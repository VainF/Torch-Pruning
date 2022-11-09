import abc
import torch
import torch.nn as nn

from .pruner import function
from ._helpers import _FlattenIndexTransform
from . import ops
import math

class ImportanceNormalizer():
    def __call__(self, group, imp):
        return imp

class StandardizationNormalizer(ImportanceNormalizer):
    def __init__(self, mode='mean_var'):
        assert mode in ('min_max', 'min_var')
        self.mode = mode

    def __call__(self, group, imp):
        if self.mode=='mean_var':
            return (imp - imp.mean()) / (1e-8+imp.std())
        elif self.mode=='min_max':
            return (imp - imp.min()) / (imp.max() - imp.min())

class ReducingNormalizer(ImportanceNormalizer):
    def __init__(self, reduction='sum'):
        self.reduction = reduction

    def __call__(self, group, imp):
        if self.reduction=='sum':
            return imp / imp.sum()
        elif self.reduction=='mean':
            return imp / imp.mean()
        elif self.reduction=='max':
            return imp / imp.max()
        elif self.reduction=='min':
            return imp / imp.min()
        else:
            raise NotImplementedError

class RelativeNormalizer():
    def __init__(self, percentage=None):
        self._k = dict()
        self.percentage = percentage

    def __call__(self, group, imp):
        m = group[0][0].target
        if m not in self._k:
            k = max(int(self.percentage * len(imp)), 1)
            self._k[m] = k
        else:
            k = self._k[m]
        #div = imp.topk(k=min(k, len(imp)), dim=0, largest=True)[0][-1]
        imp = imp / imp.topk(k=min(k, len(imp)), dim=0, largest=True)[0].mean()
        return imp


class Importance(abc.ABC):
    """ estimate the importance of PruningGroup and return an 1-D per-channel importance.
    """
    @abc.abstractclassmethod
    def __call__(self, group)-> torch.Tensor:
        raise NotImplementedError

class MagnitudeImportance(Importance):
    def __init__(self, p=1, to_group=False, group_reduction="mean", normalizer=ReducingNormalizer(reduction='mean')):
        self.p = p
        self.to_group = to_group
        self.group_reduction = group_reduction
        self.normalizer = normalizer

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        return group_imp

    @torch.no_grad()
    def __call__(self, group, **kwargs):
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

        #group_imp = torch.stack(group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return group_imp


class SaliencyImportance(MagnitudeImportance):
    def __init__(self, p=1, to_group=False, group_reduction='mean', normalizer=ReducingNormalizer(reduction='mean')):
        super().__init__(p=p, to_group=to_group, group_reduction=group_reduction, normalizer=normalizer)

    @torch.no_grad()
    def __call__(self, group, **kwargs):
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
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return -group_imp


class BNScaleImportance(MagnitudeImportance):
    def __init__(self, to_group=False, group_reduction="mean", normalizer=None):
        super().__init__(p=1, to_group=to_group, group_reduction=group_reduction, normalizer=normalizer)
    
    def __call__(self, group, **kwargs):
        group_imp = []
        for dep, _ in group:
            module = dep.target.module
            if isinstance(module, (ops.TORCH_BATCHNORM)) and module.affine:
                local_imp = torch.abs(module.weight.data)
                group_imp.append(local_imp)
                if not self.to_group:
                    return local_imp

        group_imp = torch.stack(group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return group_imp


class LAMPImportance(MagnitudeImportance):
    def __init__(self, p=2, to_group=False, group_reduction="mean", normalizer=None):
        super().__init__(p=p, to_group=to_group, group_reduction=group_reduction, normalizer=normalizer)

    @torch.no_grad()
    def __call__(self, group, **kwargs):
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
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
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
    def __init__(self, p=2, normalizer=RelativeNormalizer(percentage=0.0)):
        self.p = p
        self.normalizer = normalizer
        
    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_norm = 0
        group_size = 0

        #Get group norm
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            # Conv out_channels
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                w = layer.weight.data[idxs].flatten(1)
                local_norm = w.pow(2).sum(1)
                #print(local_norm.shape, layer, idxs, ch_groups)
                if ch_groups>1:
                    local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                group_norm+=local_norm
                #if layer.bias is not None:
                #    group_norm += layer.bias.data[idxs].pow(2)
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                w = (layer.weight).transpose(0, 1).flatten(1)
                if (
                    w.shape[0] != group_norm.shape[0]
                ):  
                    if hasattr(dep.target, 'index_transform') and isinstance(dep.target.index_transform, _FlattenIndexTransform):
                        # conv - latten
                        w = w.view(
                            group_norm.shape[0],
                            w.shape[0] // group_norm.shape[0],
                            w.shape[1],
                        ).flatten(1)
                    elif ch_groups>1 and prune_fn==function.prune_conv_in_channels and layer.groups==1:
                        # group conv
                        w = w.view(w.shape[0] // group_norm.shape[0],
                                group_norm.shape[0], w.shape[1]).transpose(0, 1).flatten(1)               
                local_norm = w.pow(2).sum(1)
                if ch_groups>1:
                    if len(local_norm)==len(group_norm):
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                group_norm += local_norm[idxs]
            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_norm = w.pow(2)
                    if ch_groups>1:
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                        local_norm = local_norm.repeat(ch_groups)
                    group_norm += local_norm

                    #b = layer.bias.data[idxs]
                    #local_norm = b.pow(2)
                    #if ch_groups>1:
                    #    local_norm = local_norm.view(ch_groups, -1).sum(0)
                    #    local_norm = local_norm.repeat(ch_groups)
                    #group_norm += local_norm
            elif prune_fn == function.prune_lstm_out_channels:
                _idxs = torch.tensor(idxs)
                local_norm = 0
                local_norm_reverse = 0
                num_layers = layer.num_layers
                expanded_idxs = torch.cat([ _idxs+i*layer.hidden_size for i in range(4) ], dim=0)
                if layer.bidirectional:
                    postfix = ['', '_reverse']
                else:
                    postfix = ['']

                local_norm+=getattr(layer, 'weight_hh_l0')[expanded_idxs].pow(2).sum(1).view(4, -1).sum(0)
                local_norm+=getattr(layer, 'weight_hh_l0')[:, _idxs].pow(2).sum(0)
                local_norm+=getattr(layer, 'weight_ih_l0')[expanded_idxs].pow(2).sum(1).view(4, -1).sum(0)
                if layer.bidirectional:
                    local_norm_reverse+=getattr(layer, 'weight_hh_l0')[expanded_idxs].pow(2).sum(1).view(4, -1).sum(0)
                    local_norm_reverse+=getattr(layer, 'weight_hh_l0')[:, _idxs].pow(2).sum(0)
                    local_norm_reverse+=getattr(layer, 'weight_ih_l0')[expanded_idxs].pow(2).sum(1).view(4, -1).sum(0)
                    local_norm = torch.cat([local_norm, local_norm_reverse], dim=0)
                group_norm += local_norm
            elif prune_fn == function.prune_lstm_in_channels:
                local_norm=getattr(layer, 'weight_ih_l0')[:, idxs].pow(2).sum(0)
                if layer.bidirectional:
                    local_norm_reverse+=getattr(layer, 'weight_ih_l0_reverse')[:, idxs].pow(2).sum(0)
                    local_norm = torch.cat([local_norm, local_norm_reverse], dim=0)
                group_norm+=local_norm
        group_imp = group_norm**(1/self.p)
        group_size = math.sqrt(group_size)
        print("="*15)
        print(group)
        print(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return group_imp 

class GroupBNScalingImportance(Importance):
    def __init__(self, p=2, normalizer=RelativeNormalizer(percentage=0.0)):
        self.p = p
        self.normalizer = normalizer
        
    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_norm = 0
        group_size = 0

        #Get group norm
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
    
            # Conv in_channels
            if prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_norm = w.pow(2)
                    if ch_groups>1:
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                        local_norm = local_norm.repeat(ch_groups)
                    group_norm += local_norm
                    group_size += ch_groups
        group_imp = group_norm**(1/self.p)
        group_size = math.sqrt(group_size)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        #group_imp = (group_imp-group_imp.mean()) / group_imp.std()
        return group_imp 



class RandomImportance(Importance):
    @torch.no_grad()
    def __call__(self, group, **kwargs):
        _, idxs = group[0]
        return torch.rand(len(idxs))

class GroupConvImportance(MagnitudeImportance):
    def __init__(self, p=2, to_group=False, group_reduction="mean", normalizer=RelativeNormalizer(percentage=0.5)):
        super().__init__(p=p, to_group=to_group, group_reduction=group_reduction, normalizer=normalizer)

    @torch.no_grad()
    def __call__(self, group, **kwargs):
        group_norm = 0
        group_size = 0
        # Get group norm
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            if prune_fn in [
                function.prune_conv_out_channels,
            ]:
                # regularize output channels
                w = layer.weight.data[idxs].flatten(1)
                group_size += w.shape[1]
                group_norm += w.abs().pow(self.p).sum(1)
        group_imp = group_norm**(1/self.p)
        #if self.normalizer is not None:
        #    group_imp = self.normalizer(group, group_imp)
        group_norm / group_size
        return group_norm


class RandomImportance(Importance):
    @torch.no_grad()
    def __call__(self, group, **kwargs):
        _, idxs = group[0]
        return torch.rand(len(idxs))