import abc
import torch
import torch.nn as nn

import typing
from .pruner import function
from .dependency import Group
from ._helpers import _FlattenIndexMapping
from . import ops
import math


class Importance(abc.ABC):
    """ Estimate the importance of a Pruning Group, and return an 1-D per-channel importance score.

        It should accept a group and a ch_groups as inputs, and return a 1-D tensor with the same length as the number of channels.
        ch_groups refer to the number of internal groups, e.g., for a 64-channel **group conv** with groups=ch_groups=4, each group has 16 channels.
        All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
        Just ignore the ch_groups if you are not sure what it is.
    """
    @abc.abstractclassmethod
    def __call__(self, group: Group, ch_groups: int=1) -> torch.Tensor: 
        raise NotImplementedError


class MagnitudeImportance(Importance):
    def __init__(self, p=2, group_reduction="mean", normalizer='mean'):
        self.p = p
        self.group_reduction = group_reduction
        self.normalizer = normalizer

    def _normalize(self, group_importance, normalizer):
        if normalizer is None:
            return group_importance
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (group_importance.max() - group_importance.min()+1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == 'gaussian':
            return (group_importance - group_importance.mean()) / (group_importance.std()+1e-8)
        else:
            raise NotImplementedError

    def _reduce(self, group_imp: typing.List[torch.Tensor], group_idxs: typing.List[typing.List[int]]):
        if len(group_imp) == 0: return group_imp
        reduced_imp = torch.zeros_like(group_imp[0])

        for i, (imp, root_idxs) in enumerate(zip(group_imp, group_idxs)):
            if self.group_reduction == "sum" or self.group_reduction == "mean":
                reduced_imp.scatter_add_(0, torch.tensor(root_idxs), imp) # accumulated importance
            elif self.group_reduction == "max": # keep the max importance
                selected_imp = torch.select(reduced_imp, 0, torch.tensor(root_idxs))
                torch.max(selected_imp, imp, out=selected_imp)
                reduced_imp.scatter_(0, torch.tensor(root_idxs), selected_imp)
            elif self.group_reduction == "prod": # product of importance
                selected_imp = torch.select(reduced_imp, 0, torch.tensor(root_idxs))
                torch.mul(selected_imp, imp, out=selected_imp)
                reduced_imp.scatter_(0, torch.tensor(root_idxs), selected_imp)
            elif self.group_reduction == 'first':
                if i == 0:
                    reduced_imp.scatter_(0, torch.tensor(root_idxs), imp)
            elif self.group_reduction == 'gate':
                if i == len(group_imp)-1:
                    reduced_imp.scatter_(0, torch.tensor(root_idxs), imp)
            elif self.group_reduction is None:
                reduced_imp = group_imp # no reduction
            else:
                raise NotImplementedError
            return reduced_imp
        
    @torch.no_grad()
    def __call__(self, group: Group, ch_groups: int=1):
        group_imp = []
        group_idxs = []
        # Get group norm
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            
            root_idxs = group[i].root_idxs
            # Conv out_channels
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                if ch_groups > 1:
                    local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                group_imp.append(local_norm)
                group_idxs.append(root_idxs)
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                is_conv_flatten_linear = False
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight).flatten(1)
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)
                if ch_groups > 1 and prune_fn == function.prune_conv_in_channels and layer.groups == 1:
                    # non-grouped conv and group convs
                    w = w.view(w.shape[0] // group_imp[0].shape[0],
                               group_imp[0].shape[0], w.shape[1]).transpose(0, 1).flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                if ch_groups > 1:
                    if len(local_norm) == len(group_imp[0]):
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                local_norm = local_norm[idxs]
                group_imp.append(local_norm)
                group_idxs.append(root_idxs)
            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_norm = w.abs().pow(self.p)
                    if ch_groups > 1:
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                        local_norm = local_norm.repeat(ch_groups)
                    # print(local_norm.shape)
                    group_imp.append(local_norm)
                    group_idxs.append(root_idxs)
        if len(group_imp) == 0:
            return None
        imp_size = len(group_imp[0])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp) == imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


class BNScaleImportance(MagnitudeImportance):
    """Learning Efficient Convolutional Networks through Network Slimming, 
    https://arxiv.org/abs/1708.06519
    """

    def __init__(self, group_reduction='mean', normalizer='mean'):
        super().__init__(p=1, group_reduction=group_reduction, normalizer=normalizer)

    def __call__(self, group, ch_groups=1):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            module = dep.target.module
            root_idxs = group[i].root_idxs
            if isinstance(module, (ops.TORCH_BATCHNORM)) and module.affine:
                local_imp = torch.abs(module.weight.data)
                if ch_groups > 1:
                    local_imp = local_imp.view(ch_groups, -1).mean(0)
                    local_imp = local_imp.repeat(ch_groups)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)
        if len(group_imp) == 0:
            return None
        group_imp = torch.stack(group_imp, dim=0)
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


class LAMPImportance(MagnitudeImportance):
    """Layer-adaptive Sparsity for the Magnitude-based Pruning,
    https://arxiv.org/abs/2010.07611
    """

    def __init__(self, p=2, group_reduction="mean", normalizer='mean'):
        super().__init__(p=p, group_reduction=group_reduction, normalizer=normalizer)

    @torch.no_grad()
    def __call__(self, group, **kwargs):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight)[:, idxs].transpose(0, 1)
                else:
                    w = (layer.weight)[idxs]
                local_imp = torch.norm(
                    torch.flatten(w, 1), dim=1, p=self.p)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight)[idxs].flatten(1)
                else:
                    w = (layer.weight)[:, idxs].transpose(0, 1).flatten(1)
                if (
                    w.shape[0] != group_imp[0].shape[0]
                ):  # for conv-flatten-linear without global pooling
                    w = w.view(
                        group_imp[0].shape[0],
                        w.shape[0] // group_imp[0].shape[0],
                        w.shape[1],
                    ).flatten(1)
                local_imp = torch.norm(w, dim=1, p=self.p)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)
            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine is not None:
                    w = (layer.weight)[idxs].view(-1, 1)
                    local_imp = torch.norm(w, dim=1, p=self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
        if len(group_imp) == 0:
            return None
        group_imp = torch.stack(group_imp, dim=0)
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return self.lamp(group_imp)

    def lamp(self, imp):
        argsort_idx = torch.argsort(imp, dim=0, descending=True).tolist()
        sorted_imp = imp[argsort_idx]
        cumsum_imp = torch.cumsum(sorted_imp, dim=0)
        sorted_imp = sorted_imp / cumsum_imp
        inversed_idx = torch.arange(len(sorted_imp))[
            argsort_idx
        ].tolist()  # [0, 1, 2, 3, ..., ]
        return sorted_imp[inversed_idx]


class RandomImportance(Importance):
    @torch.no_grad()
    def __call__(self, group, **kwargs):
        _, idxs = group[0]
        return torch.rand(len(idxs))


class GroupNormImportance(MagnitudeImportance):
    def __init__(self, p=2, normalizer='max'):
        super().__init__(p=p, group_reduction=None, normalizer=normalizer)
        self.p = p
        self.normalizer = normalizer

    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_norm = None

        # Get group norm
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            # Conv out_channels
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, 'transposed') and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                #print(local_norm.shape, layer, idxs, ch_groups)
                if ch_groups > 1:
                    local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                if group_norm is None: group_norm = local_norm
                elif group_norm.shape[0] == local_norm.shape[0]:
                    group_norm += local_norm
                # if layer.bias is not None:
                #    group_norm += layer.bias.data[idxs].pow(2)
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                is_conv_flatten_linear = False
                if hasattr(layer, 'transposed') and layer.transposed:
                    w = (layer.weight).flatten(1)
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)
                if (w.shape[0] != group_norm.shape[0]):
                    if (hasattr(dep, 'index_mapping') and isinstance(dep.index_mapping, _FlattenIndexMapping)):
                        # conv-flatten
                        w = w[idxs].view(
                            group_norm.shape[0],
                            w.shape[0] // group_norm.shape[0],
                            w.shape[1],
                        ).flatten(1)
                        is_conv_flatten_linear = True
                    elif ch_groups > 1 and prune_fn == function.prune_conv_in_channels and layer.groups == 1:
                        # non-grouped conv with group convs
                        w = w.view(w.shape[0] // group_norm.shape[0],
                                   group_norm.shape[0], w.shape[1]).transpose(0, 1).flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                if ch_groups > 1:
                    if len(local_norm) == len(group_norm):
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                if not is_conv_flatten_linear:
                    local_norm = local_norm[idxs]
                if group_norm is None: group_norm = local_norm
                elif group_norm.shape[0] == local_norm.shape[0]:
                    group_norm += local_norm
            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_norm = w.abs().pow(self.p)
                    if ch_groups > 1:
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                        local_norm = local_norm.repeat(ch_groups)
                    if group_norm is None: group_norm = local_norm
                    elif group_norm.shape[0] == local_norm.shape[0]:
                        group_norm += local_norm

            elif prune_fn == function.prune_lstm_out_channels:
                _idxs = torch.tensor(idxs)
                local_norm = 0
                local_norm_reverse = 0
                num_layers = layer.num_layers
                expanded_idxs = torch.cat(
                    [_idxs+i*layer.hidden_size for i in range(4)], dim=0)
                if layer.bidirectional:
                    postfix = ['', '_reverse']
                else:
                    postfix = ['']

                local_norm += getattr(layer, 'weight_hh_l0')[expanded_idxs].abs().pow(
                    self.p).sum(1).view(4, -1).sum(0)
                local_norm += getattr(layer,
                                      'weight_hh_l0')[:, _idxs].abs().pow(self.p).sum(0)
                local_norm += getattr(layer, 'weight_ih_l0')[expanded_idxs].abs().pow(
                    self.p).sum(1).view(4, -1).sum(0)
                if layer.bidirectional:
                    local_norm_reverse += getattr(layer, 'weight_hh_l0')[
                        expanded_idxs].abs().pow(self.p).sum(1).view(4, -1).sum(0)
                    local_norm_reverse += getattr(layer, 'weight_hh_l0')[
                        :, _idxs].abs().pow(self.p).sum(0)
                    local_norm_reverse += getattr(layer, 'weight_ih_l0')[
                        expanded_idxs].abs().pow(self.p).sum(1).view(4, -1).sum(0)
                    local_norm = torch.cat(
                        [local_norm, local_norm_reverse], dim=0)
                if group_norm is None: group_norm = local_norm
                elif group_norm.shape[0] == local_norm.shape[0]:
                    group_norm += local_norm
            elif prune_fn == function.prune_lstm_in_channels:
                local_norm = getattr(layer, 'weight_ih_l0')[
                    :, idxs].abs().pow(self.p).sum(0)
                if layer.bidirectional:
                    local_norm_reverse += getattr(layer, 'weight_ih_l0_reverse')[
                        :, idxs].abs().pow(self.p).sum(0)
                    local_norm = torch.cat(
                        [local_norm, local_norm_reverse], dim=0)
                if group_norm is None: group_norm = local_norm
                elif group_norm.shape[0] == local_norm.shape[0]:
                    group_norm += local_norm
                    
        group_imp = group_norm**(1/self.p)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


class TaylorImportance(Importance):
    def __init__(self, group_reduction="mean", normalizer='mean', multivariable=False):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.multivariable = multivariable

    def _normalize(self, group_importance, normalizer):
        if normalizer is None:
            return group_importance
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (group_importance.max() - group_importance.min()+1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == 'gaussian':
            return (group_importance - group_importance.mean()) / (group_importance.std()+1e-8)
        else:
            raise NotImplementedError

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction == 'first':
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else:
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                    dw = layer.weight.grad.data.transpose(1, 0)[
                        idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                    dw = layer.weight.grad.data[idxs].flatten(1)
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                group_imp.append(local_imp)
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight).flatten(1)[idxs]
                    dw = (layer.weight.grad).flatten(1)[idxs]
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)[idxs]
                    dw = (layer.weight.grad).transpose(0, 1).flatten(1)[idxs]
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                group_imp.append(local_imp)
            # BN
            elif prune_fn == function.prune_groupnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w*dw).abs()
                    group_imp.append(local_imp)
        if len(group_imp) == 0:
            return None
        imp_size = len(group_imp[0])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp) == imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp
