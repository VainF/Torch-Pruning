import argparse
import os
import sys

from loguru import logger
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma
from timm.loss import LabelSmoothingCrossEntropy
from timm.data import Mixup
    
OFA_HOME = os.environ["OFA_HOME"]
sys.path.append(OFA_HOME)

from ofa.imagenet_classification.networks import MobileNetV3Large
import torch_pruning as tp
from process import train_one_epoch, validate
from ft_utils import ProgressMonitor, PerformanceScoreboard
from ddp_utils import (fix_random_seed, init_distributed_training, init_logger,
    setup_print, load_data_dist)
from loss_ops import SoftTargetCrossEntropyNoneSoftmax


def load_models(net, model_path=None):
    # specify init path
    init = torch.load(model_path, map_location="cpu")["state_dict"]
    net.load_state_dict(init)


def init_dataloader(args):
    args.dataloader = {
        "dataset": "imagenet",
        "num_classes": 1000,
        "path": "/datasets/imagenet",
        "batch_size": 128,
        "workers": 16,
    }
    
    return args


def parse_args():
    arg_parser = argparse.ArgumentParser()
    # Pruning
    arg_parser.add_argument("pr", type=float)
    
    # DDP
    arg_parser.add_argument("--local_rank", default=0, type=int)
    
    # IO
    arg_parser.add_argument("--name", type=str, default="default")
    arg_parser.add_argument("--output_dir", type=str, default="exp")
    
    # Data loader
    arg_parser.add_argument("--dataset", type=str, default="imagenet")
    arg_parser.add_argument("--path", type=str, default="/datasets/imagenet")
    arg_parser.add_argument("--num_classes", type=int, default=1000)
    arg_parser.add_argument("--batch_size", type=int, default=128)
    arg_parser.add_argument("--workers", type=int, default=16)
    arg_parser.add_argument("--aug_type", type=str, default="auto")
    arg_parser.add_argument("--color_jitter", type=float, default=0.4)
    arg_parser.add_argument("--aa", type=str, default="rand-n1-m1-mstd0.5-inc1")
    arg_parser.add_argument("--train_interpolation", type=str, default="bicubic")
    arg_parser.add_argument("--reprob", type=float, default=0.01)
    arg_parser.add_argument("--remode", type=str, default="pixel")
    arg_parser.add_argument("--recount", type=int, default=1)
    arg_parser.add_argument("--mixup", action='store_true', default=True)
    
    # mixup
    arg_parser.add_argument('--cutmix', type=float, default=0.01,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    arg_parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    arg_parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    arg_parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    arg_parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    arg_parser.add_argument("--smoothing", type=float, default=0.1)
    
    # Optimizer
    arg_parser.add_argument("--opt", type=str, default="adamw")
    arg_parser.add_argument("--lr", type=float, default=1e-4)
    arg_parser.add_argument("--momentum", type=float, default=0.9)
    arg_parser.add_argument("--weight_decay", type=float, default=0.05)
    
    # Scheduler
    arg_parser.add_argument("--sched", type=str, default="cosine")
    arg_parser.add_argument("--min_lr", type=float, default=1e-6)
    arg_parser.add_argument("--decay_rate", type=float, default=0.1)
    arg_parser.add_argument("--warmup_epochs", type=int, default=5)
    arg_parser.add_argument("--warmup_lr", type=float, default=1e-6)
    arg_parser.add_argument("--decay_epochs", type=int, default=30)
    arg_parser.add_argument("--cooldown_epochs", type=int, default=10)
    
    # KD
    arg_parser.add_argument("--distillation", action='store_true', default=False)

    # Train
    arg_parser.add_argument("--epochs", type=int, default=30)
    arg_parser.add_argument("--val_cycle", type=int, default=50)
    
    # Log
    arg_parser.add_argument("--num_best_scores", type=int, default=3)
    arg_parser.add_argument("--print_freq", type=int, default=20)
    
    # Eval
    arg_parser.add_argument("--eval", action='store_true', default=False)
    
    args = arg_parser.parse_args()
    return args


def main():
    bn_momentum = 0.1
    bn_eps = 1e-5
    model = MobileNetV3Large(
        n_classes=1000,
        bn_param=(bn_momentum, bn_eps),
        dropout_rate=0,
        width_mult=1.0,
        ks=7,
        expand_ratio=6,
        depth_param=4,
    )
    args = parse_args()

    model_path = ".torch/ofa_checkpoints/ofa_D4_E6_K7"

    load_models(model, model_path)
    print(model)

    example_inputs = torch.randn(1, 3, 224, 224)

    # 1. Importance criterion
    imp = tp.importance.GroupTaylorImportance() # or GroupNormImportance(p=2), GroupHessianImportance(), etc.

    # 2. Initialize a pruner with the model and the importance criterion
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m) # DO NOT prune the final classifier!

    pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=args.pr, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
        ignored_layers=ignored_layers,
    )

    # 3. Prune the model
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    if isinstance(imp, tp.importance.GroupTaylorImportance):
        # Taylor expansion requires gradients for importance estimation
        loss = model(example_inputs).sum() # A dummy loss, please replace this line with your loss function and data!
        loss.backward() # before pruner.step()

    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print("Base MACs: %f G, Pruned MACs: %f G"%(base_macs/1e9, macs/1e9))
    print("Base Params: %f M, Pruned Params: %f M"%(base_nparams/1e6, nparams/1e6))
    print(model)

    # 4. Finetune the model
    init_distributed_training(args)
    print(f'training on world_size {dist.get_world_size()}, rank {dist.get_rank()}, local_rank {args.local_rank}')
    fix_random_seed(seed=0)
    
    output_dir = args.output_dir
    
    pymonitor = None
    if args.rank == 0:
        os.makedirs(output_dir, exist_ok=True)

        log_dir = init_logger(
            args.name, output_dir)

        with open(f"{args.output_dir}/args.json", "w") as args_file:  # dump experiment config
            json.dump(vars(args), args_file, indent=4)

        pymonitor = ProgressMonitor(logger)

    assert args.rank >= 0, 'ERROR IN RANK'
    assert args.distributed
    
    setup_print(is_master=args.rank == 0)
    
    if args.rank == 0:
        print(args)

    scaled_linear_lr = args.lr * dist.get_world_size() * args.batch_size / 512
    scaled_linear_min_lr = args.min_lr * \
        dist.get_world_size() * args.batch_size / 512
    scaled_linear_warmup_lr = args.warmup_lr * \
        dist.get_world_size() * args.batch_size / 512
        
    args.lr = scaled_linear_lr
    args.min_lr = scaled_linear_min_lr
    args.warmup_lr = scaled_linear_warmup_lr
    
    start_epoch = 0
    
    # model EMA
    model.cuda()
    
    model_ema = ModelEma(model=model, decay=0.99985, device="", resume="")
    
    # optimizer
    optimizer = create_optimizer(args, model)
    
    model_ema.ema.train()  # use training mode to track the running states
    model = DistributedDataParallel(
        model, device_ids=[args.local_rank], find_unused_parameters=True)
    
    # data loader
    train_loader, val_loader, test_loader, training_sampler = load_data_dist(args)
    
    mixup_fn = None
    if args.mixup:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    # loss
    if args.mixup:
        criterion = SoftTargetCrossEntropyNoneSoftmax()
    else:
        criterion = LabelSmoothingCrossEntropy(args.smoothing)
        
    criterion = criterion.cuda()
    
    num_epochs = args.epochs
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    
    # KD
    teacher_model = None
    distillation_loss = None
    
    teacher_path = model_path
    
    if args.distillation and not args.eval:
        teacher_model = build_teachers(args, pretrained_teacher_path)

        if args.distillation:
            distillation_loss = SoftTargetCrossEntropyNoneSoftmax().cuda()

    if start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.rank == 0:
        logger.info(('Optimizer: %s' % optimizer).replace(
            '\n', '\n' + ' ' * 11))
        logger.info('Total epoch: %d, Start epoch %d, Val cycle: %d',
                    num_epochs, start_epoch, args.val_cycle)
    
    perf_scoreboard = PerformanceScoreboard(args.num_best_scores)

    v_top1, v_top5, v_loss = 0, 0, 0
    
    if args.eval:
        top1_eval_acc = validate(
           val_loader, model, None, 0, args)

        if args.rank == 0:
            logger.info(
                f"[Eval mode] evaluation top-1 accuracy {top1_eval_acc} (%)")
        return 

    top1_eval_acc = validate(
           val_loader, model, None, 0, args)
    if args.rank == 0:
        logger.info(
            f"[Pruned] evaluation top-1 accuracy {top1_eval_acc} (%)")
    
    for epoch in range(start_epoch, num_epochs):
        if args.distributed:
            training_sampler.set_epoch(epoch)

        if args.rank == 0:
            logger.info('>>>>>>>> Epoch %3d' % epoch)

        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, lr_scheduler, epoch, pymonitor, args,
            teacher_model=teacher_model, distillation_loss=distillation_loss,
            mixup_fn=mixup_fn, model_ema=model_ema)

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1)

        top1_eval_acc = validate(
            val_loader, model, None, epoch, args)

        if args.rank == 0:
            logger.info(
                f"Evaluation accuracy (min subnet) [{epoch}/{num_epochs}] {top1_eval_acc}")
            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)

            save_checkpoint(epoch, 'supernet', model, {
                'top1': v_top1, 'top5': v_top5}, is_best, args.name, output_dir, optimizer=optimizer, model_ema=model_ema)

            if epoch % 10 == 0:
                save_checkpoint(epoch, 'supernet', model, {
                    'top1': v_top1, 'top5': v_top5}, False, args.name + f'_{epoch}epochs_', output_dir, optimizer=optimizer, model_ema=model_ema)

    if args.rank == 0: 
        logger.info('Program completed successfully ... exiting ...')

if __name__ == "__main__":
    main()
