# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Code source: https://github.com/microsoft/Moonlit/blob/main/ElasticViT/process.py
import logging
import math
import operator
import time
import random
import torch
import torch.nn.functional as F
import numpy as np
from ft_utils import AverageMeter
# from timm.utils.agc import adaptive_clip_grad
import torch.nn as nn
from torch.distributed import get_world_size, all_reduce, barrier

__all__ = ['train_one_epoch', 'validate', 'PerformanceScoreboard']

logger = logging.getLogger()


def bn_cal(model, train_loader, args, arch=None, num_batches=100, mixup_fn=None):
    model.eval()

    for _, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.training = True
            module.momentum = None

            module.reset_running_stats()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if batch_idx > num_batches:
            break

        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        if mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)

        model(inputs)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(model, train_loader, val_loader, args, mixup_fn):
    with torch.no_grad():
        logger.info("start batch-norm layer calibration...")
        bn_cal(model, train_loader, args, num_batches=64 *
               (256//args.batch_size), mixup_fn=mixup_fn)
        logger.info("finish batch-norm layer calibration...")
    acc1_val, _, _ = validate(val_loader, model, None, 0, None, args, None, )
    return round(acc1_val, 2)


def update_meter(meter, loss, acc1, acc5, size, batch_time, world_size):

    barrier()
    r_loss = loss.clone().detach()
    all_reduce(r_loss)
    r_loss /= world_size

    meter['loss'].update(r_loss.item(), size)
    meter['batch_time'].update(batch_time)


def teacher_inference(teacher_model, inputs, T=0.2):
    with torch.no_grad():
        if isinstance(teacher_model, list):
            teacher_outputs_0 = (teacher_model[0](inputs) / T).softmax(dim=-1)
            teacher_outputs_1 = (teacher_model[1](inputs) / T).softmax(dim=-1)
            teacher_outputs = teacher_outputs_0 / 2. + teacher_outputs_1 / 2.
        else:
            teacher_outputs = teacher_model(inputs).softmax(dim=-1)

    return teacher_outputs


def compute_dist_loss(labels, outputs, criterion, teacher_outputs=None, distill_criterion=None, multi_teachers=False, ALPHA=.5):
    distill_loss = None
    if teacher_outputs and distill_criterion:
        distill_loss = distill_criterion(outputs, teacher_outputs)

    if multi_teachers:
        return distill_loss
    else:
        if distill_loss:
            return ALPHA * criterion(outputs, labels) + (1-ALPHA) * distill_loss
        else:
            return criterion(outputs, labels)


def train_one_epoch(train_loader, model, criterion, optimizer, lr_scheduler, epoch, pymonitor, args, distillation_loss, teacher_model,
                    mixup_fn, model_ema, record_one_epoch=False, hard_distillation=False, force_random=False):
    meters = {
        'loss': AverageMeter(),
        'batch_time': AverageMeter()
    }

    total_sample = len(train_loader.sampler)
    batch_size = args.batch_size

    steps_per_epoch = math.ceil(total_sample / batch_size)
    steps_per_epoch = torch.tensor(steps_per_epoch).to(args.device)
    all_reduce(steps_per_epoch)
    steps_per_epoch = int(steps_per_epoch.item() // get_world_size())

    if args.rank == 0:
        logger.info('Training: %d samples (%d per mini-batch)',
                    total_sample, batch_size)

    num_updates = epoch * steps_per_epoch
    seed = num_updates
    model.train()
    model_without_ddp = model.module

    if teacher_model:
        if isinstance(teacher_model, list):
            for tm in teacher_model:
                tm.eval()
        else:
            teacher_model.eval()

    teacher_outputs = None
    multi_teachers = isinstance(teacher_model, list)

    for batch_idx, (original_inputs, original_targets) in enumerate(train_loader):
        original_inputs = original_inputs.to(args.device)
        original_targets = original_targets.to(args.device)

        optimizer.zero_grad()
        seed = seed + 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if mixup_fn is not None:
            inputs, targets = original_inputs.clone(), original_targets.clone()
            inputs, targets = mixup_fn(inputs, targets)
        else:
            inputs, targets = original_inputs, original_targets

        if teacher_model is not None:
            teacher_outputs = teacher_inference(
                teacher_model=teacher_model, inputs=inputs)
        
        start_time = time.time()
        outputs = model(inputs)

        loss = None
        
        loss = compute_dist_loss(labels=targets, outputs=outputs, teacher_outputs=teacher_outputs, criterion=criterion, distill_criterion=distillation_loss,
                                    multi_teachers=multi_teachers)
        
        if loss is None:
            raise NotImplementedError

        loss.backward()
        update_meter(meters, loss, None, None, inputs.size(
            0), time.time() - start_time, args.world_size)

        # adaptive_clip_grad(model.parameters(), 0.1, norm_type=2.0)
        optimizer.step()

        num_updates += 1

        if lr_scheduler is not None:
            lr_scheduler.step_update(
                num_updates=num_updates, metric=meters['loss'].avg)

        torch.cuda.synchronize()

        if model_ema is not None:
            model_ema.update(model)

        if args.rank == 0 and (batch_idx + 1) % args.print_freq == 0:
            pymonitor.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                'Loss': meters['loss'],
                'BatchTime': meters['batch_time'],
                'LR': optimizer.param_groups[0]['lr'],
                'GPU memory': round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
            })
            logger.info(
                "--------------------------------------------------------------------------------------------------------------")
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    if 'top1' in meters.keys():
        return meters['top1'].avg, meters['top5'].avg, meters['loss'].avg
    else:
        return meters['loss'].avg


def validate_single(data_loader, model, criterion, epoch, args):
    meters = {
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'top5': AverageMeter(),
        'batch_time': AverageMeter()
    }

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            start_time = time.time()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            print(acc1)
            update_meter(meters, loss, acc1, acc5, inputs.size(
                0), time.time() - start_time, args.world_size)

    return meters['top1'].avg, meters['top5'].avg, meters['loss'].avg


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch
