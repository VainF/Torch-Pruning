import sys, os
from tkinter import TRUE

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from functools import partial

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
parser.add_argument("--total_epochs", type=int, default=100)
parser.add_argument("--speed_up", type=float, default=2)
parser.add_argument("--soft_rank", type=float, default=0.5)
parser.add_argument(
    "--lr_decay_milestones",
    default="60,80",
    type=str,
    help="milestones for learning rate decay",
)
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--restore", type=str, default=None)
parser.add_argument("--pruning_steps", type=int, default=100)
parser.add_argument("--save_dir", type=str, default="run")


# Pruning options
parser.add_argument("--sparsity", type=float, default=1.0)
parser.add_argument("--method", type=str, default=None)
parser.add_argument("--reg", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--global_pruning", action="store_true", default=False)

args = parser.parse_args()


def eval(model, test_loader, device=None):
    correct = 0
    total = 0
    loss = 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img, target = img.to(device), target.to(device)
            out = model(img)
            loss += F.cross_entropy(out, target, reduction="sum")
            pred = out.max(1)[1]
            correct += (pred == target).sum()
            total += len(target)
    return (correct / total).item(), (loss / total).item()

def iterate_dataset(
    model,
    train_loader,
    pruner,
    device=None,
):
    pruner.reset()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    for i, (img, target) in enumerate(train_loader):
        model.zero_grad()
        img, target = img.to(device), target.to(device)
        out = model(img)
        loss = F.cross_entropy(out, target)
        pruner.update_importance(loss)

def train_model(
    model,
    epochs,
    train_loader,
    test_loader,
    # For pruning only
    pruner=None,
    regularize=False,
    save_to=None,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4 if regularize==False else 0,
    )
    milestones = [int(ms) for ms in args.lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)
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
                gnorm_list, scale_list = pruner.regularize(model, loss)
                #if i==0:
                #    for k, (g, s) in enumerate(zip(gnorm_list, scale_list)):
                #        print("Group %d"%(k))
                #        print(g)
                #        print(g/g.max())
                #        print()
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
        acc, val_loss = eval(model, test_loader, device=device)
        args.logger.info(
            "Epoch {:d}/{:d}, Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}".format(
                epoch, epochs, acc, val_loss, optimizer.param_groups[0]["lr"]
            )
        )
        if best_acc < acc:
            os.makedirs(args.save_dir, exist_ok=True)
            if args.mode == "prune" and save_to is not None:
                torch.save(
                   model.state_dict(),
                   save_to,
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
    user_defined_parameters = (
        [model.pos_embedding, model.cls_token] if "vit" in args.model else None
    )
    example_inputs = torch.randn(1, 3, 32, 32).to(args.device)
    sparsity_learning = False
    data_dependency = False
    if args.method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "lamp":
        imp = tp.importance.LAMPImportance(p=2, to_group=False)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "slim":
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance(to_group=False)
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == 'sensitivity':
        data_dependency = True
        imp = tp.importance.SaliencyImportance(to_group=False)
        pruner_entry = partial(tp.pruner.SaliencyPruner, global_pruning=args.global_pruning)
    elif args.method == 'group_sensitivity':
        data_dependency = True
        imp = tp.importance.SaliencyImportance(to_group=True, reduction='sum', normalize=True, soft_rank=args.soft_rank )
        pruner_entry = partial(tp.pruner.SaliencyPruner, global_pruning=args.global_pruning)
    elif args.method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2, to_group=True, soft_rank=args.soft_rank, normalize=True)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
    elif args.method == "group_sl":
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2, to_group=True, soft_rank=args.soft_rank, normalize=True)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    
    
    # TODO: Delete
    elif args.method == "group_lasso_oneshot":
        sparsity_learning = False
        imp = tp.importance.GroupNormImportance(p=1, to_group=True, soft_rank=args.soft_rank, normalize=True)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "group_lasso":
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2, to_group=True, soft_rank=args.soft_rank, normalize=True)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    else:
        raise NotImplementedError
    args.data_dependency = data_dependency
    args.sparsity_learning = sparsity_learning
    ignored_layers = []
    layer_ch_sparsity = {}
    # ignore output layers
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
            ignored_layers.append(m)
    
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        pruning_steps=args.pruning_steps,
        ch_sparsity=args.sparsity,
        layer_ch_sparsity=layer_ch_sparsity,
        ignored_layers=ignored_layers,
        user_defined_parameters=user_defined_parameters,
    )
    return pruner


def main():
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.mode == "prune":
        exp_id = "{dataset}-{model}-{method}".format(
            dataset=args.dataset,
            model=args.model,
            method=args.method,
        )
        if args.global_pruning:
            exp_id+="-global"
        args.save_dir = os.path.join(args.save_dir, args.mode, exp_id)
        log_file = "{}/{}_{}_{}.txt".format(
            args.save_dir, args.dataset, args.method, args.model
        )
    elif args.mode == "pretrain":
        args.save_dir = os.path.join(args.save_dir, args.mode)
        log_file = "{}/{}_{}.txt".format(args.save_dir, args.dataset, args.model)
    elif args.mode == "test":
        log_file = None
    logger_name = "{}_{}_{}".format(args.dataset, args.method, args.model)
    if args.global_pruning:
        logger_name+='_global'
    args.logger = tools.utils.get_logger(logger_name, output=log_file)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            model.load_state_dict(loaded)
        args.logger.info("Loading model from {restore}".format(restore=args.restore))

    model = model.to(args.device)
    ######################################################
    # Training / Pruning / Testing
    if args.mode == "train":
        train_model(
            model=model,
            epochs=args.epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            pruner=None,
        )
    elif args.mode == "prune":
        model.eval()
        # States before pruning
        ori_macs, ori_size = tp.utils.count_macs_and_params(
            model,
            input_size=(1, 3, 32, 32),
            device=args.device,
        )
        ori_acc, ori_val_loss = eval(model, test_loader, device=args.device)
        pruner = get_pruner(model, args)

        
        if args.data_dependency:
            iterate_dataset(model=model, train_loader=train_loader, pruner=pruner, device=args.device)

        if args.sparsity_learning:
            reg_pth = "reg_{}_{}_{}.pth".format(
                           args.dataset, args.model, args.method
                       )
            reg_pth = os.path.join( os.path.join(args.save_dir, reg_pth) )
            args.logger.info("regularizing...")
            train_model(
                model,
                args.total_epochs,
                train_loader,
                test_loader,
                pruner=pruner,
                regularize=True,
                save_to = reg_pth,
            )
            model.load_state_dict( torch.load( reg_pth, map_location=args.device) )
        
        # 2. Pruning
        model.eval()
        current_speed_up = 1
        while current_speed_up < args.speed_up:
            pruner.step()
            pruned_macs, pruned_size = tp.utils.count_macs_and_params(
                model,
                input_size=(1, 3, 32, 32),
                device=args.device,
            )
            current_speed_up = float(ori_macs) / pruned_macs

        args.logger.info(model)
        pruned_macs, pruned_size = tp.utils.count_macs_and_params(
            model,
            input_size=(1, 3, 32, 32),
            device=args.device,
        )
        pruned_acc, pruned_val_loss = eval(model, test_loader, device=args.device)
        args.logger.info("Sparsity: %.2f" % (args.sparsity))
        args.logger.info(
            "Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(
                ori_size / 1e6, pruned_size / 1e6, pruned_size / ori_size * 100
            )
        )
        args.logger.info(
            "MACs: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
                ori_macs / 1e6,
                pruned_macs / 1e6,
                pruned_macs / ori_macs * 100,
                ori_macs / pruned_macs,
            )
        )
        args.logger.info("Acc: {:.4f} => {:.4f}".format(ori_acc, pruned_acc))
        args.logger.info(
            "Val Loss: {:.4f} => {:.4f}".format(ori_val_loss, pruned_val_loss)
        )

        # 3. finetuning
        train_model(
            model,
            args.total_epochs,
            train_loader,
            test_loader,
            pruner,
            device=args.device,
        )
    elif args.mode == "finetune":
        train_model(
            model,
            args.total_epochs,
            train_loader,
            test_loader,
            pruner,
            device=args.device,
        )
    elif args.mode == "test":
        model.eval()
        args.logger.info("Load model from {}".format(args.restore))
        macs, params = tp.utils.count_macs_and_params(
            model, input_size=(1, 3, 32, 32), device=args.device
        )
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("MACs: {:.2f} M".format(macs / 1e6))
        acc, val_loss = eval(model, test_loader)
        args.logger.info("Acc: {:.4f} Val Loss: {:.4f}\n".format(acc, val_loss))


if __name__ == "__main__":
    main()
