import argparse
import os
import sys

from loguru import logger
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import timm
from timm import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma
from timm.loss import LabelSmoothingCrossEntropy
from timm.data import Mixup
    
import torch_pruning as tp
from process import train_one_epoch, validate
from ft_utils import ProgressMonitor, PerformanceScoreboard
from ddp_utils import (fix_random_seed, init_distributed_training, init_logger,
    setup_print, load_data_dist)
from loss_ops import SoftTargetCrossEntropyNoneSoftmax
from checkpoint import save_checkpoint, load_checkpoint


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
    # DDP
    arg_parser.add_argument("--local_rank", default=0, type=int)
    
    # IO
    arg_parser.add_argument("--pt_filepath", type=str, default=None)
    arg_parser.add_argument("--output_dir", type=str, default="exp")
    arg_parser.add_argument("--exp_name", type=str, default="test")
    
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
    arg_parser.add_argument("--teacher_path", type=str, default=None)
    
    # EMA
    arg_parser.add_argument("--ema", action="store_true", default=False)

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
    args = parse_args()
    example_inputs = torch.randn(1, 3, 224, 224)

    init_distributed_training(args)
    logger.info(f'training on world_size {dist.get_world_size()}, rank {dist.get_rank()}, local_rank {args.local_rank}')
    fix_random_seed(seed=0)
    
    output_dir = f"{args.output_dir}/{args.exp_name}"
    model = torch.jit.load(args.pt_filepath)
    
    pymonitor = None
    if args.rank == 0:
        os.makedirs(output_dir, exist_ok=True)

        logger.add(f"{output_dir}/log.txt")

        with open(f"{args.output_dir}/args.json", "w") as args_file:  # dump experiment config
            json.dump(vars(args), args_file, indent=4)

        pymonitor = ProgressMonitor(logger)

    assert args.rank >= 0, 'ERROR IN RANK'
    assert args.distributed
    
    setup_print(is_master=args.rank == 0)
    
    if args.rank == 0:
        print(args)

    # Finetune the model
    scaled_linear_lr = args.lr * dist.get_world_size() * args.batch_size / 512
    scaled_linear_min_lr = args.min_lr * \
        dist.get_world_size() * args.batch_size / 512
    scaled_linear_warmup_lr = args.warmup_lr * \
        dist.get_world_size() * args.batch_size / 512
        
    args.lr = scaled_linear_lr
    args.min_lr = scaled_linear_min_lr
    args.warmup_lr = scaled_linear_warmup_lr
    
    start_epoch = 0
    
    model.cuda()
    
    # model EMA
    model_ema = None
    if args.ema:
        model_ema = ModelEma(model=model, decay=0.99985, device="", resume="")
        model_ema.ema.train()  # use training mode to track the running states
    
    # optimizer
    optimizer = create_optimizer(args, model)
    
    model = DistributedDataParallel(
        model, device_ids=[args.local_rank])
    
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

    # ------------- auto resume -------------
    chkp_file = self.training_config.resume_path if (self.training_config.resume_path is not None and os.path.exists(self.training_config.resume_path)) else os.path.join(output_dir, self.training_config.name + '_checkpoint.pth.tar')
    if os.path.exists(chkp_file):
        print("load checkpoint from", chkp_file)
        super_model, start_epoch, _ = load_checkpoint(
            super_model, chkp_file=chkp_file, strict=True, lean=self.training_config.resume_lean, optimizer=optimizer if not self.training_config.eval else None)
        if self.training_config.model_ema:
            model_ema.ema = deepcopy(super_model)
        # Update super weights
        self.tailor.tir.update_weights(super_model.named_parameters())
    else:
        assert not self.training_config.eval
    
    num_epochs = self.training_config.epochs
    if int(start_epoch) == int(num_epochs):
        # training has finished
        return    
    
    # KD
    teacher_model = None
    distillation_loss = None
    
    teacher_path = args.teacher_path
    
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

            save_checkpoint(epoch, model, {
                'top1': v_top1, 'top5': v_top5}, is_best, args.exp_name, output_dir, optimizer=optimizer, model_ema=model_ema, output_torchscript=True)

            if epoch % 10 == 0:
                save_checkpoint(epoch, model, {
                    'top1': v_top1, 'top5': v_top5}, False, args.exp_name + f'_{epoch}epochs_', output_dir, optimizer=optimizer, model_ema=model_ema, output_torchscript=True)

    if args.rank == 0: 
        logger.info('Program completed successfully ... exiting ...')

if __name__ == "__main__":
    main()
