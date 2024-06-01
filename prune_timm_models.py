import argparse
import os
import sys

from loguru import logger
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

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
    # Pruning
    arg_parser.add_argument("pr", type=float)
    
    # IO
    arg_parser.add_argument("--name", type=str, default="default")
    arg_parser.add_argument("--output_dir", type=str, default="exp")
    arg_parser.add_argument("--exp_name", type=str, default="test")
    args = arg_parser.parse_args()
    return args


def main():
    args = parse_args()
    model = create_model(
        args.name,
        pretrained=True)
    
    example_inputs = torch.randn(1, 3, 224, 224)

    fix_random_seed(seed=0)
    
    output_dir = f"{args.output_dir}/{args.exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    
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
    logger.info(f"Base MACs: {base_macs/1e9} G, Pruned MACs: {macs/1e9} G")
    logger.info(f"Base Params: {base_nparams/1e6} M, Pruned Params: {nparams/1e6} M")
    logger.info(model)

    save_checkpoint(None, model, output_dir=output_dir, name="pruned", output_torchscript=True)

    logger.info('Program completed successfully ... exiting ...')

if __name__ == "__main__":
    main()
