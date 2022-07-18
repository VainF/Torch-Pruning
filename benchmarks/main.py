import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import torch_pruning as tp
import registry
import tools

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", type=str, required=True, choices=["pretrain", "prune", "test"]
)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--dataset", type=str, default="cifar100")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--total_epochs", type=int, default=120)
parser.add_argument(
    "--lr_decay_milestones",
    default="60,90",
    type=str,
    help="milestones for learning rate decay",
)
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--restore", type=str, default=None)
parser.add_argument("--pruning_steps", type=int, default=1)
parser.add_argument("--save_dir", type=str, default="run")


# Pruning options
parser.add_argument("--sparsity", type=float, default=0.4)
parser.add_argument("--local", action="store_true", default=False)
parser.add_argument("--method", type=str, default=None)


args = parser.parse_args()


def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred == target).sum()
            total += len(target)
    return correct / total


def train_model(
    model,
    epochs,
    train_loader,
    test_loader,
    # For pruning
    pruner,
    regularize=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    milestones = [int(ms) for ms in args.lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)
    model.train()
    best_acc = -1
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            if regularize:
                pruner.regularize(model)
            optimizer.step()
            if i % 10 == 0 and args.verbose:
                args.logger.info(
                    "Epoch {:d}/{:d}, iter {:d}/{:d}, loss={:.4f}, lr={:.4f}".format(
                        epoch,
                        epochs,
                        i,
                        len(train_loader),
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )
        model.eval()
        acc = eval(model, test_loader)
        args.logger.info(
            "Epoch {:d}/{:d}, Acc={:.4f}, lr={:.4f}".format(
                epoch, epochs, acc, optimizer.param_groups[0]["lr"]
            )
        )
        if best_acc < acc:
            os.makedirs(args.save_dir, exist_ok=True)
            if args.mode == "prune":
                torch.save(
                    model,
                    os.path.join(
                        args.save_dir,
                        "{}_{}_{}_step{}.pth".format(
                            args.dataset, args.model, args.method, pruner.current_step
                        ),
                    ),
                )
            elif args.mode == "train":
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.save_dir, "{}_{}.pth".format(args.dataset, args.model)
                    ),
                )
            best_acc = acc
        scheduler.step()
    args.logger.info("Best Acc=%.4f" % (best_acc))


def get_pruner(model, args):
    model.cpu()
    user_defined_parameters = (
        [model.pos_embedding, model.cls_token] if "vit" in args.model else None
    )
    example_inputs = torch.randn(1, 3, 32, 32)
    requires_reg = False
    if args.method == "ours":
        imp = tp.importance.StrcuturalImportance(p=1, reduction="sum")
        pruner_entry = tp.pruner.LocalMagnitudePruner
    elif args.method == "sreg":
        requires_reg = True
        imp = tp.importance.MagnitudeImportance(p=1, reduction="sum")
        pruner_entry = tp.pruner.LocalStructrualRegularizedPruner
    elif args.method == "dropout":
        requires_reg = True
        imp = tp.importance.RandomImportance()
        pruner_entry = tp.pruner.StructrualDropoutPruner
    elif args.method in "l1":
        imp = tp.importance.MagnitudeImportance(p=1, local=True)
        pruner_entry = tp.pruner.LocalMagnitudePruner
    elif args.method in "l1_global":
        imp = tp.importance.MagnitudeImportance(p=1, local=True)
    elif args.method in "l1_group":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = tp.pruner.LocalMagnitudePruner
    elif args.method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = tp.pruner.LocalMagnitudePruner
    elif args.method == "lamp_global":
        imp = tp.importance.LAMPImportance()
        pruner_entry = tp.pruner.GlobalMagnitudePruner
    elif args.method == "lamp":
        imp = tp.importance.LAMPImportance()
        pruner_entry = tp.pruner.LocalMagnitudePruner
    elif args.method == "slim":
        requires_reg = True
        imp = tp.importance.BNScaleImportance()
    elif args.method == "slim_group":
        requires_reg = True
        imp = tp.importance.BNScaleImportance(group_level=True)
        pruner_entry = tp.pruner.LocalBNScalePruner
    elif args.method == "slim_global":
        requires_reg = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = tp.pruner.GlobalBNScalePruner
    args.requires_reg = requires_reg
    ignored_layers = []
    layer_ch_sparsity = {}
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
            ignored_layers.append(m)
        # if isinstance(m, torch.nn.Conv2d) and m.kernel_size[0]==1:
        #    ignored_layers.append(m)

    # if 'resnet56' in args.model:
    #    ignored_layers.append(model.conv1)
    #    layer_ch_sparsity[model.layer1] = 0.75
    #    layer_ch_sparsity[model.layer2] = 0.75
    #    layer_ch_sparsity[model.layer3] = 0.32
    #    args.sparsity = 0

    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        total_steps=args.pruning_steps,
        ch_sparsity=args.sparsity,
        layer_ch_sparsity=layer_ch_sparsity,
        ignored_layers=ignored_layers,
        user_defined_parameters=user_defined_parameters,
    )
    return pruner


def main():
    if args.mode == "prune":
        exp_id = "{dataset}-{model}-{method}-{time}".format(
            dataset=args.dataset,
            model=args.model,
            method=args.method,
            time=time.asctime().replace(" ", "_"),
        )
        log_file = "{}/{}_{}_{}.txt".format(
            args.save_dir, args.dataset, args.method, args.model
        )
        args.save_dir = os.path.join(args.save_dir, args.mode, exp_id)
    elif args.mode == "pretrain":
        args.save_dir = os.path.join(args.save_dir, args.mode)
        log_file = "{}/{}_{}.txt".format(args.save_dir, args.dataset, args.model)
    elif args.mode == "test":
        log_file = None
    args.logger = tools.utils.get_logger(args.mode, output=log_file)

    # Model & Dataset
    num_classes, train_dst, val_dst = registry.get_dataset(
        args.dataset, data_root="data"
    )
    model = registry.get_model(args.model, num_classes=num_classes)
    train_loader = torch.utils.data.DataLoader(
        train_dst,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_dst, batch_size=args.batch_size, num_workers=4
    )
    args.num_classes = num_classes

    for k, v in tools.utils.flatten_dict(vars(args)).items():  # print args
        args.logger.info("%s: %s" % (k, v))

    if args.restore is not None:
        loaded = torch.load(args.restore, map_location="cpu")
        if isinstance(loaded, nn.Module):
            model = loaded
        else:
            model.load_state_dict(loaded["state_dict"])
        args.logger.info("Loading model from {restore}".format(restore=args.restore))

    ######################################################
    # Training / Pruning / Testing
    if args.mode == "train":
        train_model(
            model=model,
            epochs=args.total_epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            pruner=None,
        )
    elif args.mode == "prune":
        model.eval()
        # States before pruning
        ori_macs, ori_size = tp.utils.count_macs_and_params(
            model, input_size=(1, 3, 32, 32)
        )
        ori_acc = eval(model, test_loader)

        pruner = get_pruner(model, args)
        for step in range(pruner.total_steps):
            args.logger.info("Pruning step %d" % (step))

            if args.method == "dropout":
                pruner.register_structural_dropout(model)

            if args.requires_reg:
                train_model(
                    model,
                    args.total_epochs // 3,
                    train_loader,
                    test_loader,
                    pruning_step=pruner.current_step,
                    pruner=pruner,
                    regularize=True,
                )

            if args.method == "dropout":
                pruner.remove_structural_dropout()

            model.cpu().eval()
            pruner.step()
            args.logger.info(model)

            pruned_macs, pruned_size = tp.utils.count_macs_and_params(
                model, input_size=(1, 3, 32, 32)
            )
            pruned_acc = eval(model, test_loader)

            args.logger.info("Sparsity: %.2f" % (args.sparsity))
            args.logger.info(
                "Params: {:.2f} M => {:.2f} M ({:.2f} %%)".format(
                    ori_size / 1e6, pruned_size / 1e6, pruned_size / ori_size * 100
                )
            )
            args.logger.info(
                "MACs: {:.2f} M => {:.2f} M ({:.2f} %%)".format(
                    ori_macs / 1e6, pruned_macs / 1e6, pruned_macs / ori_macs * 100
                )
            )
            args.logger.info(
                "Acc: {:.4f} M => {:.4f} M ({:.4f} %%)".format(
                    ori_acc, pruned_acc, pruned_acc / ori_acc * 100
                )
            )

            train_model(
                model,
                args.total_epochs,
                train_loader,
                test_loader,
                pruner,
            )
    elif args.mode == "test":
        model.eval()
        args.logger.info("Load model from {}".format(args.restore))
        params = tools.utils.get_n_params(model)
        macs = tools.utils.get_n_macs(model, img_size=(32, 32))
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("MACs: {:.2f} M".format(macs / 1e6))
        acc = eval(model, test_loader)
        args.logger.info("Acc: {:.4f}\n".format(acc))


if __name__ == "__main__":
    main()
