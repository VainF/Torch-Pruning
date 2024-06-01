
import os
import random

from loguru import logger
import numpy as np

import torch
import torch.distributed as dist
import torch.utils.data
import torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from timm.data import create_transform


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_distributed_training(args):
    
    args.device = "cuda"

    args.world_size = 1
    args.rank = 0
    args.distributed = True

    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])

    args.device = 'cuda:%d' % args.local_rank
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)


def init_logger(exp_name, output_dir):
    log_path = f"{output_dir}/{exp_name}.log"
    logger.add(log_path)
    

def setup_print(is_master):
        
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
    
def load_data_dist(cfg, searching_set=False):
    assert cfg.dataset == 'imagenet'
    normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    traindir = os.path.join(cfg.path, 'train')
    valdir = os.path.join(cfg.path, 'val')
    print("Train dir:", traindir)

    aug_type = getattr(cfg, 'aug_type', 'none')
    if aug_type == 'auto':
        transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=cfg.color_jitter,
            auto_augment=cfg.aa,
            interpolation=cfg.train_interpolation,
            re_prob=cfg.reprob,
            re_mode=cfg.remode,
            re_count=cfg.recount,
        )

        train_set = datasets.ImageFolder(
            traindir,
            transform=transform
        )
    else:
        train_set = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=False, drop_last=True,
        sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=cfg.batch_size*2, shuffle=False,
        num_workers=cfg.workers, pin_memory=False, drop_last=False,
        sampler=val_sampler)

    if searching_set:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(cfg.path, 'search'), transforms.Compose([
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=cfg.batch_size*2, shuffle=False,
            num_workers=cfg.workers, pin_memory=False, drop_last=False)
    else:
        val_loader = test_loader

    return train_loader, val_loader, test_loader, train_sampler