import torch
import torch.nn as nn
from copy import deepcopy
from functools import reduce
from operator import mul

__all__=['prune_conv', 'prune_related_conv', 'prune_linear', 'prune_related_linear', 'prune_batchnorm', 'prune_prelu', 'prune_group_conv']

def prune_group_conv(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool=True, dry_run: bool=False):
    """Prune `filters` for the convolutional layer, e.g. [256 x 128 x 3 x 3] => [192 x 128 x 3 x 3]

    Args:
        - layer: a convolution layer.
        - idxs: pruning index.
    """
    if layer.groups>1:
         assert layer.groups==layer.in_channels and layer.groups==layer.out_channels, "only group conv with in_channel==groups==out_channels is supported"
    
    idxs = list(set(idxs))
    num_pruned = len(idxs) * reduce(mul, layer.weight.shape[1:]) + (len(idxs) if layer.bias is not None else 0)
    if dry_run: 
        return layer, num_pruned
    if not inplace:
        layer = deepcopy(layer)
    keep_idxs = [idx for idx in range(layer.out_channels) if idx not in idxs]
    layer.out_channels = layer.out_channels-len(idxs)
    layer.in_channels = layer.in_channels-len(idxs)
    layer.groups = layer.groups-len(idxs)
    layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
    if layer.bias is not None:
        layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
    return layer, num_pruned

def prune_conv(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool=True, dry_run: bool=False):
    """Prune `filters` for the convolutional layer, e.g. [256 x 128 x 3 x 3] => [192 x 128 x 3 x 3]

    Args:
        - layer: a convolution layer.
        - idxs: pruning index.
    """
    idxs = list(set(idxs))
    num_pruned = len(idxs) * reduce(mul, layer.weight.shape[1:]) + (len(idxs) if layer.bias is not None else 0)
    if dry_run: 
        return layer, num_pruned

    if not inplace:
        layer = deepcopy(layer)
    
    keep_idxs = [idx for idx in range(layer.out_channels) if idx not in idxs]
    layer.out_channels = layer.out_channels-len(idxs)
    layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
    if layer.bias is not None:
        layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
    return layer, num_pruned

def prune_related_conv(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool=True, dry_run: bool=False):
    """Prune `kernels` for the related (affected) convolutional layer, e.g. [256 x 128 x 3 x 3] => [256 x 96 x 3 x 3]

    Args:
        layer: a convolutional layer.
        idxs: pruning index.
    """
    idxs = list(set(idxs))
    num_pruned = len(idxs) *  layer.weight.shape[0] * reduce(mul ,layer.weight.shape[2:])
    if dry_run: 
        return layer, num_pruned
    if not inplace:
        layer = deepcopy(layer)
    keep_idxs = [i for i in range(layer.in_channels) if i not in idxs]
    layer.in_channels = layer.in_channels - len(idxs)
    layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs]) 
    # no bias pruning because it does not change the output size
    return layer, num_pruned

def prune_linear(layer: nn.modules.linear.Linear, idxs: list, inplace: list=True, dry_run: list=False):
    """Prune neurons for the fully-connected layer, e.g. [256 x 128] => [192 x 128]
    
    Args:
        layer: a fully-connected layer.
        idxs: pruning index.
    """
    num_pruned = len(idxs)*layer.weight.shape[1] + (len(idxs) if layer.bias is not None else 0)
    if dry_run:
        return layer, num_pruned

    if not inplace:
        layer = deepcopy(layer)
    keep_idxs = [i for i in range(layer.out_features) if i not in idxs]
    layer.out_features = layer.out_features-len(idxs)
    layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
    if layer.bias is not None:
        layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
    return layer, num_pruned

def prune_related_linear(layer: nn.modules.linear.Linear, idxs: list, inplace: list=True, dry_run: list=False):
    """Prune weights for the related (affected) fully-connected layer, e.g. [256 x 128] => [256 x 96]
    
    Args:
        layer: a fully-connected layer.
        idxs: pruning index.
    """
    num_pruned = len(idxs) *  layer.weight.shape[0]
    if dry_run:
        return layer, num_pruned
    
    if not inplace:
        layer = deepcopy(layer)
    keep_idxs = [i for i in range(layer.in_features) if i not in idxs]
    layer.in_features = layer.in_features-len(idxs)
    layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
    return layer, num_pruned

def prune_batchnorm(layer: nn.modules.batchnorm._BatchNorm, idxs: list, inplace: bool=True, dry_run: bool=False ):
    """Prune batch normalization layers, e.g. [128] => [64]

    Args:
        layer: a batch normalization layer.
        idxs: pruning index.
    """

    num_pruned = len(idxs)* ( 2 if layer.affine else 1)
    if dry_run:
        return layer, num_pruned
    
    if not inplace:
        layer = deepcopy(layer)

    keep_idxs = [i for i in range(layer.num_features) if i not in idxs] 
    layer.num_features = layer.num_features-len(idxs)
    layer.running_mean = layer.running_mean.data.clone()[keep_idxs]
    layer.running_var = layer.running_var.data.clone()[keep_idxs]
    if layer.affine:
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
    return layer, num_pruned

def prune_prelu(layer: nn.PReLU, idxs: list, inplace: bool=True, dry_run: bool=False):
    """Prune PReLU layers, e.g. [128] => [64] or [1] => [1] (no pruning if prelu has only 1 parameter)
    
    Args:
        layer: a PReLU layer.
        idxs: pruning index.
    """
    num_pruned = 0 if layer.num_parameters==1 else len(idxs)
    if dry_run:
        return layer, num_pruned
    if not inplace:
        layer = deepcopy(layer)
    if layer.num_parameters==1: return layer, num_pruned
    keep_idxs = [i for i in range(layer.num_parameters) if i not in idxs]
    layer.num_parameters = layer.num_parameters-len(idxs)
    layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
    return layer, num_pruned 


