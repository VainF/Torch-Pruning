r"""PyTorch Detection Training.
To run in a multi-gpu environment, use the distributed launcher::
    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU
The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.
On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3
Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os, sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


from engine.utils.detection_utils import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from engine.utils.detection_utils import utils

from engine.utils.detection_utils.coco_utils import get_coco, get_coco_kp
from engine.utils.detection_utils.engine import evaluate, train_one_epoch
from engine.utils.detection_utils.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
from engine.utils.detection_utils.transforms import SimpleCopyPaste

import torch_pruning as tp 
from functools import partial
from engine.utils import count_ops_and_params, MagnitudeRecover

def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))


def get_dataset(name, image_set, transform, data_path):
    paths = {"coco": (data_path, get_coco, 91), "coco_kp": (data_path, get_coco_kp, 2)}
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, args):
    if train:
        return presets.DetectionPresetTrain(data_augmentation=args.data_augmentation)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval()


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    # pruning parameters
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--method", type=str, default='l1')
    parser.add_argument("--global-pruning", default=False, action="store_true")
    parser.add_argument("--target-flops", type=float, default=2.0, help="GFLOPs of pruned model")
    parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--max-ch-sparsity", default=1.0, type=float, help="maximum channel sparsity")
    parser.add_argument("--sl-epochs", type=int, default=None)
    parser.add_argument("--sl-resume", type=str, default=None)
    parser.add_argument("--sl-lr", default=None, type=float, help="learning rate")
    parser.add_argument("--sl-lr-step-size", default=None, type=int, help="milestones for learning rate decay")
    parser.add_argument("--sl-lr-warmup-epochs", default=None, type=int, help="the number of epochs to warmup (default: 0)")

    return parser

def prune_to_target_flops(pruner, model, target_flops, example_inputs):
    model.eval()
    ori_ops, _ = count_ops_and_params(model, example_inputs=example_inputs)
    pruned_ops = ori_ops
    while pruned_ops / 1e9 > target_flops:
        pruner.step()
        if 'vit' in args.model:
            model.hidden_dim = model.conv_proj.out_channels
        pruned_ops, _ = count_ops_and_params(model, example_inputs=example_inputs)
        
    return pruned_ops

def get_pruner(model, example_inputs, args):
    unwrapped_parameters = (
        [model.encoder.pos_embedding, model.class_token] if "vit" in args.model else None
    )
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
    elif args.method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2, normalizer=tp.importance.RelativeNormalizer(args.soft_keeping_ratio))
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
    elif args.method == "group_sl":
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2, normalizer=tp.importance.RelativeNormalizer(args.soft_keeping_ratio))
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    else:
        raise NotImplementedError
    args.data_dependency = data_dependency
    args.sparsity_learning = sparsity_learning
    ignored_layers = []
    ch_sparsity_dict = {}
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            ignored_layers.append(m)
    round_to = None
    if 'vit' in args.model:
        round_to = model.encoder.layers[0].num_heads
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=100,
        ch_sparsity=1.0,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=args.max_ch_sparsity,
        ignored_layers=ignored_layers,
        round_to=round_to,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(True, args), args.data_path)
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(False, args), args.data_path)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

        train_collate_fn = copypaste_collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    print("Creating model")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    model = torchvision.models.detection.__dict__[args.model](
        weights=args.weights, weights_backbone=args.weights_backbone, num_classes=num_classes, **kwargs
    )
    model.eval()
    print("="*16)
    print(model)
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    example_inputs = (images, targets)
    base_ops, base_params = count_ops_and_params(model, example_inputs=example_inputs)
    print("Params: {:.4f} M".format(base_params / 1e6))
    print("ops: {:.4f} G".format(base_ops / 1e9))
    print("="*16)
    if args.prune:
        pruner = get_pruner(model, example_inputs=example_inputs, args=args)
        if args.sparsity_learning:
            if args.sl_resume:
                print("Loading sparse model from {}...".format(args.sl_resume))
                model.load_state_dict( torch.load(args.sl_resume, map_location='cpu')['model'] )
            else:
                print("Sparsifying model...")
                if args.sl_lr is None: args.sl_lr = args.lr
                if args.sl_lr_step_size is None: args.sl_lr_step_size = args.lr_step_size
                if args.sl_lr_warmup_epochs is None: args.sl_lr_warmup_epochs = args.lr_warmup_epochs
                if args.sl_epochs is None: args.sl_epochs = args.epochs
                train(model, args.sl_epochs, 
                                        lr=args.sl_lr, lr_step_size=args.sl_lr_step_size, lr_warmup_epochs=args.sl_lr_warmup_epochs, 
                                        train_sampler=train_sampler, data_loader=data_loader, data_loader_test=data_loader_test, 
                                        device=device, args=args, regularizer=pruner.regularize, state_dict_only=True)
                #model.load_state_dict( torch.load('regularized_{:.4f}_best.pth'.format(args.reg), map_location='cpu')['model'] )
                #utils.save_on_master(
                #    model_without_ddp.state_dict(),
                #    os.path.join(args.output_dir, 'regularized-{:.4f}.pth'.format(args.reg)))

        model = model.to('cpu')
        print("Pruning model...")
        prune_to_target_flops(pruner, model, args.target_flops, example_inputs)
        pruned_ops, pruned_size = count_ops_and_params(model, example_inputs=example_inputs)
        print("="*16)
        print("After pruning:")
        print(model)
        print("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(base_params / 1e6, pruned_size / 1e6, pruned_size / base_params * 100))
        print("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(base_ops / 1e9, pruned_ops / 1e9, pruned_ops / base_ops * 100, base_ops / pruned_ops))
        print("="*16)

    train(model=model, epochs=args.epochs, lr=args.lr, lr_steps=args.lr_steps, train_sampler=train_sampler, data_loader=data_loader, data_loader_test=data_loader_test,
          device=device, args=args, regularizer=None)


def train(
    model, 
    epochs, 
    lr, lr_steps, 
    train_sampler, data_loader, data_loader_test, 
    device, args, regularizer=None):
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    weight_decay = args.weight_decay if regularizer is None else 0
    norm_weight_decay = args.norm_weight_decay if regularizer is None else 0

    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [norm_weight_decay, weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=args.momentum,
            weight_decay=weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        torch.backends.cudnn.deterministic = True
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, epochs, args.print_freq, scaler,  regularizer)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if regularizer is None:
                checkpoint['arch'] = model_without_ddp
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            #utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            #utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, prefix+"latest.pth"))
        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)