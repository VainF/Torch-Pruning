import abc
import torch
import torch.nn as nn

from .pruner import function
from ._helpers import _FlattenIndexMapping
from . import ops
import math

class Importance(abc.ABC):
    """ estimate the importance of a Pruning Group, and return an 1-D per-channel importance score.
    """
    @abc.abstractclassmethod
    def __call__(self, group)-> torch.Tensor:
        raise NotImplementedError

class MagnitudeImportance(Importance):
    def __init__(self, p=2, group_reduction="mean", normalizer=None):
        self.p = p
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
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
        #Get group norm
        #print(group)
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
                local_norm = w.abs().pow(self.p).sum(1)
                if ch_groups>1:
                    local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                group_imp.append(local_norm)

            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                is_conv_flatten_linear = False
                if isinstance(layer, nn.ConvTranspose2d):
                    w = (layer.weight).flatten(1)  
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)     
                if ch_groups>1 and prune_fn==function.prune_conv_in_channels and layer.groups==1:
                    # non-grouped conv and group convs
                    w = w.view(w.shape[0] // group_imp[0].shape[0],
                            group_imp[0].shape[0], w.shape[1]).transpose(0, 1).flatten(1)       
                local_norm = w.abs().pow(self.p).sum(1)
                if ch_groups>1:
                    if len(local_norm)==len(group_imp[0]):
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                local_norm = local_norm[idxs]
                group_imp.append(local_norm)
            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_norm = w.abs().pow(self.p)
                    if ch_groups>1:
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                        local_norm = local_norm.repeat(ch_groups)
                    #print(local_norm.shape)
                    group_imp.append(local_norm)
        if len(group_imp)==0:
            return None
        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)>min_imp_size and len(imp)%min_imp_size==0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp)==min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return group_imp


class BNScaleImportance(MagnitudeImportance):
    """Learning Efficient Convolutional Networks through Network Slimming, 
    https://arxiv.org/abs/1708.06519
    """
    def __init__(self, group_reduction='mean', normalizer=None):
        super().__init__(p=1, group_reduction=group_reduction, normalizer=normalizer)
    
    def __call__(self, group, ch_groups=1):
        group_imp = []
        for dep, _ in group:
            module = dep.target.module
            if isinstance(module, (ops.TORCH_BATCHNORM)) and module.affine:
                local_imp = torch.abs(module.weight.data)
                if ch_groups>1:
                    local_imp = local_imp.view(ch_groups, -1).mean(0)
                    local_imp = local_imp.repeat(ch_groups)
                group_imp.append(local_imp)
        if len(group_imp)==0:
            return None
        group_imp = torch.stack(group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return group_imp


class LAMPImportance(MagnitudeImportance):
    """Layer-adaptive Sparsity for the Magnitude-based Pruning,
    https://arxiv.org/abs/2010.07611
    """
    def __init__(self, p=2, group_reduction="mean", normalizer=None):
        super().__init__(p=p, group_reduction=group_reduction, normalizer=normalizer)

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
                local_imp = torch.norm(
                    torch.flatten(w, 1), dim=1, p=self.p)
                group_imp.append(local_imp)

            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
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

            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine is not None:
                    w = (layer.weight)[idxs].view(-1, 1)
                    local_imp = torch.norm(w, dim=1, p=self.p)
                    group_imp.append(local_imp)
        if len(group_imp)==0:
            return None
        group_imp = torch.stack(group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
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

class GroupNormImportance(Importance):
    def __init__(self, p=2, normalizer=None):
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
                local_norm = w.abs().pow(self.p).sum(1)
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
                is_conv_flatten_linear = False
                if isinstance(layer, nn.ConvTranspose2d):
                    w = (layer.weight).flatten(1)  
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)             
                if (w.shape[0] != group_norm.shape[0]):  
                    if (hasattr(dep, 'index_mapping') and isinstance(dep.index_mapping, _FlattenIndexMapping)):
                        #conv-flatten
                        w = w[idxs].view(
                            group_norm.shape[0],
                            w.shape[0] // group_norm.shape[0],
                            w.shape[1],
                        ).flatten(1)
                        is_conv_flatten_linear = True
                    elif ch_groups>1 and prune_fn==function.prune_conv_in_channels and layer.groups==1:
                        # non-grouped conv with group convs
                        w = w.view(w.shape[0] // group_norm.shape[0],
                                group_norm.shape[0], w.shape[1]).transpose(0, 1).flatten(1)           
                local_norm = w.abs().pow(self.p).sum(1)
                if ch_groups>1:
                    if len(local_norm)==len(group_norm):
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                if not is_conv_flatten_linear:
                    local_norm = local_norm[idxs]
                group_norm += local_norm
            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_norm = w.abs().pow(self.p)
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

                local_norm+=getattr(layer, 'weight_hh_l0')[expanded_idxs].abs().pow(self.p).sum(1).view(4, -1).sum(0)
                local_norm+=getattr(layer, 'weight_hh_l0')[:, _idxs].abs().pow(self.p).sum(0)
                local_norm+=getattr(layer, 'weight_ih_l0')[expanded_idxs].abs().pow(self.p).sum(1).view(4, -1).sum(0)
                if layer.bidirectional:
                    local_norm_reverse+=getattr(layer, 'weight_hh_l0')[expanded_idxs].abs().pow(self.p).sum(1).view(4, -1).sum(0)
                    local_norm_reverse+=getattr(layer, 'weight_hh_l0')[:, _idxs].abs().pow(self.p).sum(0)
                    local_norm_reverse+=getattr(layer, 'weight_ih_l0')[expanded_idxs].abs().pow(self.p).sum(1).view(4, -1).sum(0)
                    local_norm = torch.cat([local_norm, local_norm_reverse], dim=0)
                group_norm += local_norm
            elif prune_fn == function.prune_lstm_in_channels:
                local_norm=getattr(layer, 'weight_ih_l0')[:, idxs].abs().pow(self.p).sum(0)
                if layer.bidirectional:
                    local_norm_reverse+=getattr(layer, 'weight_ih_l0_reverse')[:, idxs].abs().pow(self.p).sum(0)
                    local_norm = torch.cat([local_norm, local_norm_reverse], dim=0)
                group_norm+=local_norm
        group_imp = group_norm**(1/self.p)
        group_imp = group_imp / group_imp.max()
        return group_imp 