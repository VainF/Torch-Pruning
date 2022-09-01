import abc
from glob import glob
import torch
import torch.nn as nn
from . import functional
import random
from ._helpers import _FlattenIndexTransform

def rescale(x):
    return (x - x.min(dim=1, keepdim=True)[0]) / (x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0])

class Importance:
    pass

class MagnitudeImportance(Importance):
    def __init__(self, p=1, to_group=False, normalize=False, reduction="mean"):
        self.p = p
        self.to_group = to_group
        self.reduction = reduction
        self.normalize = normalize

    @torch.no_grad()
    def __call__(self, group):
        global_imp = []
        for dep, idxs in group:
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w = (layer.weight)[idxs].flatten(1)
                local_imp = torch.norm(w, dim=1, p=self.p)
                global_imp.append(local_imp)

            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w = (layer.weight).transpose(0, 1).flatten(1)[idxs]
                if (
                    w.shape[0] != global_imp[0].shape[0]
                ):  # for conv-flatten-linear without global pooling
                    if (
                        w.shape[0] % global_imp[0].shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w = w.view(
                        global_imp[0].shape[0],
                        w.shape[0] // global_imp[0].shape[0],
                        w.shape[1],
                    ).flatten(1)
                local_imp = torch.norm(w, dim=1, p=self.p) 
                global_imp.append(local_imp)

            elif prune_fn == functional.prune_batchnorm:
                if layer.affine is not None:
                    w = (layer.weight)[idxs].view(-1, 1)
                    local_imp = torch.norm(w, dim=1, p=self.p)
                    global_imp.append(local_imp)

            if not self.to_group:  # we only consider the first pruned layer
                return local_imp

        global_imp = torch.stack(global_imp, dim=0)
        if self.normalize:
            global_imp = rescale(global_imp)
        if self.reduction == "sum":
            return global_imp.sum(dim=0)
        elif self.reduction == "mean":
            return global_imp.mean(dim=0)
        elif self.reduction == "max":
            return global_imp.max(dim=0)[0]
        elif self.reduction == "prod":
            return torch.prod(global_imp, dim=0)

class RandomImportance(Importance):
    @torch.no_grad()
    def __call__(self, group):
        _, idxs = group[0]
        return torch.randn((len(idxs),))
        

class SaliencyImportance(Importance):
    def __init__(self, p=1, to_group=False, normalize=False, soft_rank=0.1, reduction="mean"):
        self.p = p
        self.to_group = to_group
        self.reduction = reduction
        self.normalize = normalize
        self.soft_rank = soft_rank

    @torch.no_grad()
    def __call__(self, group):
        global_imp = []
        for dep, idxs in group:
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w_dw = (layer.weight * layer.weight.grad)[idxs].flatten(1)
                local_imp = w_dw.sum(1).abs() #torch.norm(w_dw, dim=1)
                global_imp.append(local_imp)
            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w_dw = (layer.weight * layer.weight.grad)[:, idxs].transpose(0, 1).flatten(1)
                local_imp = w_dw.sum(1).abs() #torch.norm(w_dw, dim=1)
                global_imp.append(local_imp)
            elif prune_fn == functional.prune_batchnorm:
                if layer.affine is not None:
                    w_dw = (layer.weight * layer.weight.grad)[idxs].view(-1, 1)
                    local_imp = w_dw.sum(1).abs() #torch.norm(w_dw, dim=1)
                    global_imp.append(local_imp)

            if not self.to_group:  # we only consider the first pruned layer
                return -local_imp

        global_imp = torch.stack(global_imp, dim=0)
        if self.reduction == "sum":
            global_imp = global_imp.sum(dim=0)
        elif self.reduction == "mean":
            global_imp = global_imp.mean(dim=0)
        elif self.reduction == "max":
            global_imp = global_imp.max(dim=0)[0]
        
        if self.normalize:
            k = max(int(self.soft_rank * len(global_imp)), 1)
            global_imp = global_imp / global_imp.topk(k=k, dim=0, largest=True)[0][-1]
        return -global_imp

class HessianImportance(Importance):
    def __init__(self) -> None:
        pass


class BNScaleImportance(Importance):
    def __init__(self, to_group=False, reduction="mean"):
        self.to_group = to_group
        self.reduction = reduction

    def __call__(self, group):
        global_imp = []
        for dep, idxs in group:
            # Conv-BN
            module = dep.target.module
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and module.affine:
                imp = torch.abs(module.weight.data)
                global_imp.append(imp)
                if not self.to_group:
                    return imp

        global_imp = torch.stack(global_imp, dim=0)
        if self.reduction == "sum":
            return global_imp.sum(dim=0)
        elif self.reduction == "mean":
            return global_imp.mean(dim=0)
        elif self.reduction == "max":
            return global_imp.max(dim=0)[0]


class LAMPImportance(Importance):
    def __init__(self, p=2, to_group=False, reduction="mean"):
        self.p = p
        self.to_group = to_group
        self.reduction = reduction

    @torch.no_grad()
    def __call__(self, group):
        global_imp = []
        for dep, idxs in group:
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w = (layer.weight)[idxs]
                local_imp = self.lamp(torch.norm(torch.flatten(w, 1), dim=1, p=self.p))
                global_imp.append(local_imp)

            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w = (layer.weight)[:, idxs].transpose(0, 1).flatten(1)
                if (
                    w.shape[0] != global_imp[0].shape[0]
                ):  # for conv-flatten-linear without global pooling
                    if (
                        w.shape[0] % global_imp[0].shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w = w.view(
                        global_imp[0].shape[0],
                        w.shape[0] // global_imp[0].shape[0],
                        w.shape[1],
                    ).flatten(1)
                local_imp = self.lamp(torch.norm(w, dim=1, p=self.p))

                global_imp.append(local_imp)

            elif prune_fn == functional.prune_batchnorm:
                if layer.affine is not None:
                    w = (layer.weight)[idxs].view(-1, 1)
                    local_imp = self.lamp(torch.norm(w, dim=1, p=self.p))
                    global_imp.append(local_imp)

            if not self.to_group:  # we only consider the first pruned layer
                return local_imp

        global_imp = torch.stack(global_imp, dim=0)
        if self.reduction == "sum":
            return global_imp.sum(dim=0)
        elif self.reduction == "mean":
            return global_imp.mean(dim=0)
        elif self.reduction == "max":
            return global_imp.max(dim=0)[0]

    def lamp(self, imp):
        argsort_idx = torch.argsort(imp, dim=0, descending=True).tolist()
        sorted_imp = imp[argsort_idx]
        cumsum_imp = torch.cumsum(sorted_imp, dim=0)
        sorted_imp = sorted_imp / cumsum_imp
        inversed_idx = torch.arange(len(sorted_imp))[
            argsort_idx
        ].tolist()  # [0, 1, 2, 3, ..., ]
        return sorted_imp[inversed_idx]


class GroupNormImportance(Importance):
    def __init__(self, p=2, to_group=False, soft_rank=0.5, normalize=False, reduction="mean"):
        self.p = p
        self.to_group = to_group
        self.reduction = reduction
        self.normalize = normalize
        self.soft_rank = soft_rank
        assert p in (1, 2)

    @torch.no_grad()
    def __call__(self, group):

        group_norm = 0
        group_size = 0

        # Get group norm
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                # regularize output channels
                w = layer.weight.data[idxs].flatten(1)
                group_size += w.shape[1]
                group_norm += w.abs().pow(self.p).sum(1)
                #if layer.bias is not None:
                #    group_norm += layer.bias[idxs].pow(2)
            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                # regularize input channels
                w = (layer.weight)[:, idxs].transpose(0, 1).flatten(1)
                if (
                    w.shape[0] != group_norm.shape[0]
                ):  # for conv-flatten 
                    if (
                        w.shape[0] % group_norm.shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    if hasattr(dep.target, 'index_transform') and isinstance(dep.target.index_transform, _FlattenIndexTransform):
                        w = w.view(
                        group_norm.shape[0],
                        w.shape[0] // group_norm.shape[0],
                        w.shape[1],
                    ).flatten(1)
                    else:
                        w = w.view( w.shape[0] // group_norm.shape[0], group_norm.shape[0], w.shape[1] ).transpose(0, 1).flatten(1)
                group_size += w.shape[1]
                group_norm += w.abs().pow(self.p).sum(1)
            elif prune_fn == functional.prune_batchnorm:
                # regularize BN
                if layer.affine is not None:
                    w = layer.weight.data[idxs]
                    group_norm += w.abs().pow(self.p)
                    group_size += 1
        group_norm = group_norm**(1/self.p)
        if self.normalize:
            k = max(int(self.soft_rank * len(group_norm)), 1)
            group_norm = group_norm / group_norm.topk(k=k, dim=0, largest=True)[0][-1]
        return group_norm


class MagnitudeImportance(Importance):
    def __init__(self, p=1, to_group=False, normalize=False, reduction="mean"):
        self.p = p
        self.to_group = to_group
        self.reduction = reduction
        self.normalize = normalize

    @torch.no_grad()
    def __call__(self, group):
        global_imp = []
        for dep, idxs in group:
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w = (layer.weight)[idxs]
                local_imp = torch.norm(torch.flatten(w, 1), dim=1, p=self.p)
                global_imp.append(local_imp)

            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w = (layer.weight).transpose(0, 1).flatten(1)[idxs]
                if (
                    w.shape[0] != global_imp[0].shape[0]
                ):  # for conv-flatten-linear without global pooling
                    if (
                        w.shape[0] % global_imp[0].shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w = w.view(
                        global_imp[0].shape[0],
                        w.shape[0] // global_imp[0].shape[0],
                        w.shape[1],
                    ).flatten(1)
                local_imp = torch.norm(w, dim=1, p=self.p)
                global_imp.append(local_imp)

            elif prune_fn == functional.prune_batchnorm:
                if layer.affine is not None:
                    w = (layer.weight)[idxs].view(-1, 1)
                    local_imp = torch.norm(w, dim=1, p=self.p)
                    global_imp.append(local_imp)

            if not self.to_group:  # we only consider the first pruned layer
                return local_imp

        global_imp = torch.stack(global_imp, dim=0)
        if self.normalize:
            global_imp = rescale(global_imp)
        if self.reduction == "sum":
            return global_imp.sum(dim=0)
        elif self.reduction == "mean":
            return global_imp.mean(dim=0)
        elif self.reduction == "max":
            return global_imp.max(dim=0)[0]
        elif self.reduction == "prod":
            return torch.prod(global_imp, dim=0)

class RandomImportance(Importance):
    @torch.no_grad()
    def __call__(self, group):
        _, idxs = group[0]
        return torch.rand(len(idxs))