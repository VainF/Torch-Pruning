import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
from copy import deepcopy
from functools import reduce
from operator import mul

__all__=['prune_conv', 'prune_related_conv', 'prune_linear', 'prune_related_linear', 'prune_batchnorm', 'prune_prelu']

def prune_conv(layer: nn.modules.conv._ConvNd, idxs: Sequence[int], inplace: bool =True, dry_run: bool = False):
    """Prune specified ``filters`` of the convolution layer.

    **Parameters:
        - layer: a convolution layer.
        - idxs: index of pruned filters.
    """
    num_pruned_parameters = len(idxs) * reduce(mul ,layer.weight.shape[1:]) + (len(idxs) if layer.bias else 0)
    if dry_run: 
        return layer, num_pruned_parameters

    if not inplace:
        layer = deepcopy(layer)
    num_pruned = len(idxs)
    keep_idxs = [i for i in range(layer.out_channels) if i not in idxs]
    layer.out_channels = layer.out_channels-num_pruned

    layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
    if layer.bias is not None:
        layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
    return layer, num_pruned_parameters

def prune_related_conv(layer: nn.modules.conv._ConvNd, idxs: Sequence[int], inplace: bool=True, dry_run: bool = False):
    """Prune specified ``kernels`` of the related convolution layer.

    **Parameters:
        - layer: a convolution layer.
        - idxs: index of pruned kernels.
    """
    num_pruned_parameters = len(idxs) *  layer.weight.shape[0] * reduce(mul ,layer.weight.shape[2:])
    if dry_run: 
        return layer, num_pruned_parameters

    if not inplace:
        layer = deepcopy(layer)
    num_pruned = len(idxs)
    keep_idxs = [i for i in range(layer.in_channels) if i not in idxs]
    layer.in_channels = layer.in_channels - num_pruned
    layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs]) 
    # no bias pruning because it does not change the output size
    return layer, num_pruned_parameters

def prune_linear(layer: nn.Linear, idxs: Sequence[int], inplace: bool=True, dry_run: bool = False):
    """Prune specified neurons of a linear layer.
    
    **Parameters:
        - layer: a convolution layer.
        - idxs: index of pruned kernels.
    """
    
    num_pruned_parameters = len(idxs)*layer.weight.shape[1] + (len(idxs) if layer.bias is not None else 0)
    if dry_run:
        return layer, num_pruned_parameters

    if not inplace:
        layer = deepcopy(layer)
    num_pruned = len(idxs)
    keep_idxs = [i for i in range(layer.out_features) if i not in idxs]
    layer.out_features = layer.out_features-num_pruned

    layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
    if layer.bias is not None:
        layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
    return layer, num_pruned_parameters

def prune_related_linear(layer: nn.Linear, idxs: Sequence[int], inplace: bool=True, dry_run: bool = False):
    """Prune specified weights of the related fully-connected layer.
    
    **Parameters:
        - layer: a convolution layer.
        - idxs: index of pruned kernels.
    """
    num_pruned_parameters = len(idxs) *  layer.weight.shape[0]
    if dry_run:
        return layer, num_pruned_parameters
    
    if not inplace:
        layer = deepcopy(layer)
    num_pruned = len(idxs)
    keep_idxs = [i for i in range(layer.in_features) if i not in idxs]
    layer.in_features = layer.in_features-num_pruned

    layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
    return layer, num_pruned_parameters

def prune_batchnorm(layer: nn.modules.batchnorm._BatchNorm, idxs: Sequence[int], inplace: bool=True, dry_run: bool = False ):
    """Prune batch normalization layers
    
    **Parameters:**
        - **layer**: a batch normalization layer.
        - **idxs**: indexs of pruned weights.
    """
    num_pruned_parameters = len(idxs)* ( 2 if layer.affine else 1)
    if dry_run:
        return layer, num_pruned_parameters
    
    if not inplace:
        layer = deepcopy(layer)
    num_pruned = len(idxs)
    keep_idxs = [i for i in range(layer.num_features) if i not in idxs] 

    layer.num_features = layer.num_features-num_pruned
    layer.running_mean = layer.running_mean.data.clone()[keep_idxs]
    layer.running_var = layer.running_var.data.clone()[keep_idxs]
    if layer.affine:
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
    return layer, num_pruned_parameters

def prune_prelu(layer: nn.PReLU, idxs: Sequence[int], inplace: bool=True, dry_run: bool = False):
    num_pruned_parameters = 0 if layer.num_parameters==1 else len(idxs)
    if dry_run:
        return layer, num_pruned_parameters
    
    if not inplace:
        layer = deepcopy(layer)
    if layer.num_parameters==1: return layer, num_pruned_parameters

    num_pruned = len(idxs)
    keep_idxs = [i for i in range(layer.num_parameters) if i not in idxs]
    layer.num_parameters = layer.num_parameters-num_pruned
    layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
    
    return layer, num_pruned_parameters 


