import abc
import torch
import torch.nn as nn

from .pruner import function
from ._helpers import _FlattenIndexTransform
from . import ops

class NormalizerCollection():

    @staticmethod
    def min_max_rescale(x):
        return (x - x.min()) / (x.max() - x.min())

    @staticmethod
    def sum_normalizer(x):
        return x / x.sum()
    
    @staticmethod
    def max_normalizer(x):
        return x / x.max()


class Importance(abc.ABC):
    """ Importance takes a PruningGroup as input and returns the 1-D per-channel importance for the whole group
    """
    @abc.abstractclassmethod
    def __call__(self, group)-> torch.Tensor:
        # for dep, idxs in group:
        #     module = dep.target.module
        #     pruning_fn = dep.handler
        #     estimate the importance here
        #     ...
        raise NotImplementedError


class MagnitudeImportance(Importance):
    def __init__(self, p=1, to_group=False, normalizer=None, reduction="mean"):
        self.p = p
        self.to_group = to_group
        self.reduction = reduction
        self.normalizer = normalizer

    def reduce_and_normalize(self, group_imp):
        if self.reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)

        if self.normalizer is not None:
            group_imp = self.normalizer(group_imp)
        return group_imp

    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        
        for dep, idxs in group:
            layer = dep.target.module
            prune_fn = dep.handler
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
                function.prune_depthwise_conv_out_channels,
            ]:
                w = (layer.weight)[idxs].flatten(1)
                local_imp = torch.norm(w, dim=1, p=self.p)
                group_imp.append(local_imp)
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
                function.prune_depthwise_conv_in_channels,
            ]:
                w = (layer.weight).transpose(0, 1).flatten(1)[idxs]
                if (
                    w.shape[0] != group_imp[0].shape[0]
                ):  # for conv-flatten-linear without global pooling
                    if (
                        w.shape[0] % group_imp[0].shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w = w.view(
                        group_imp[0].shape[0],
                        w.shape[0] // group_imp[0].shape[0],
                        w.shape[1],
                    ).flatten(1)
                local_imp = torch.norm(w, dim=1, p=self.p)
                group_imp.append(local_imp)
            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine is not None:
                    w = (layer.weight)[idxs].view(-1, 1)
                    local_imp = torch.norm(w, dim=1, p=self.p)
                    group_imp.append(local_imp)

            # return local importance of the first layer if to_group=False
            if not self.to_group:  
                return local_imp

        group_imp = torch.stack(group_imp, dim=0)
        group_imp = self.reduce_and_normalize(group_imp)
        return group_imp


class SaliencyImportance(MagnitudeImportance):
    def __init__(self, p=1, to_group=False, normalizer=False, soft_rank=0.1, reduction="mean"):
        self.p = p
        self.to_group = to_group
        self.reduction = reduction
        self.normalizer = normalizer
        self.soft_rank = soft_rank

    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        for dep, idxs in group:
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                w_dw = (layer.weight * layer.weight.grad)[idxs].flatten(1)
                local_imp = w_dw.sum(1).abs()  # torch.norm(w_dw, dim=1)
                group_imp.append(local_imp)
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                w_dw = (layer.weight *
                        layer.weight.grad)[:, idxs].transpose(0, 1).flatten(1)
                local_imp = w_dw.sum(1).abs()  # torch.norm(w_dw, dim=1)
                group_imp.append(local_imp)
            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine is not None:
                    w_dw = (layer.weight * layer.weight.grad)[idxs].view(-1, 1)
                    local_imp = w_dw.sum(1).abs()  # torch.norm(w_dw, dim=1)
                    group_imp.append(local_imp)

            if not self.to_group:  # we only consider the first pruned layer
                return -local_imp

        group_imp = torch.stack(group_imp, dim=0)
        group_imp = self.reduce_and_normalize(group_imp)
        return -group_imp


class BNScaleImportance(MagnitudeImportance):
    def __init__(self, to_group=False, normalizer=None, reduction="mean"):
        self.to_group = to_group
        self.reduction = reduction
        self.normalizer = normalizer

    def __call__(self, group):
        group_imp = []
        for dep, idxs in group:
            module = dep.target.module
            if isinstance(module, (ops.TORCH_BATCHNORM)) and module.affine:
                local_imp = torch.abs(module.weight.data)
                group_imp.append(local_imp)
                if not self.to_group:
                    return local_imp

        group_imp = torch.stack(group_imp, dim=0)
        group_imp = self.reduce_and_normalize(group_imp)
        return group_imp


class LAMPImportance(Importance):
    def __init__(self, p=2, to_group=False, normalizer=None, reduction="mean"):
        self.p = p
        self.to_group = to_group
        self.reduction = reduction
        self.normalizer = normalizer

    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        for dep, idxs in group:
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                w = (layer.weight)[idxs]
                local_imp = self.lamp(torch.norm(
                    torch.flatten(w, 1), dim=1, p=self.p))
                group_imp.append(local_imp)

            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                w = (layer.weight)[:, idxs].transpose(0, 1).flatten(1)
                if (
                    w.shape[0] != group_imp[0].shape[0]
                ):  # for conv-flatten-linear without global pooling
                    if (
                        w.shape[0] % group_imp[0].shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w = w.view(
                        group_imp[0].shape[0],
                        w.shape[0] // group_imp[0].shape[0],
                        w.shape[1],
                    ).flatten(1)
                local_imp = self.lamp(torch.norm(w, dim=1, p=self.p))

                group_imp.append(local_imp)

            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine is not None:
                    w = (layer.weight)[idxs].view(-1, 1)
                    local_imp = self.lamp(torch.norm(w, dim=1, p=self.p))
                    group_imp.append(local_imp)

            if not self.to_group:  # we only consider the first pruned layer
                return local_imp

        group_imp = torch.stack(group_imp, dim=0)
        group_imp = self.reduce_and_normalize(group_imp)
        return group_imp

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
    def __init__(self, p=2, to_group=False, soft_rank=0.5, normalizer=False, reduction="mean"):
        self.p = p
        self.to_group = to_group
        self.reduction = reduction
        self.normalizer = normalizer
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
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                # regularize output channels
                w = layer.weight.data[idxs].flatten(1)
                group_size += w.shape[1]
                group_norm += w.abs().pow(self.p).sum(1)
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
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
                        w = w.view(w.shape[0] // group_norm.shape[0],
                                   group_norm.shape[0], w.shape[1]).transpose(0, 1).flatten(1)
                group_size += w.shape[1]
                group_norm += w.abs().pow(self.p).sum(1)
            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine is not None:
                    w = layer.weight.data[idxs]
                    group_norm += w.abs().pow(self.p)
                    group_size += 1
        group_norm = group_norm**(1/self.p)
        
        if self.normalizer:
            group_norm = self.normalizer(group_norm)
        return group_norm


class RandomImportance(Importance):
    @torch.no_grad()
    def __call__(self, group):
        _, idxs = group[0]
        return torch.rand(len(idxs))
