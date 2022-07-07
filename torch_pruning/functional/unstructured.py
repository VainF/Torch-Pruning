import torch
import torch.nn as nn
from copy import deepcopy

__all__=['mask_weight', 'mask_bias']

def _mask_weight_hook(module, input):
    if hasattr(module, 'weight_mask'):
        module.weight.data *= module.weight_mask

def _mask_bias_hook(module, input):
    if module.bias is not None and hasattr(module, 'bias_mask'):
        module.bias.data *= module.bias_mask

def mask_weight(layer, mask, inplace=True):
    """Unstructed pruning for convolution layer

    Args:
        layer: a convolution layer.
        mask: 0-1 mask.
    """
    if not inplace:
        layer = deepcopy(layer)
    if mask.shape != layer.weight.shape:
        return layer
    mask = torch.tensor( mask, dtype=layer.weight.dtype, device=layer.weight.device, requires_grad=False )
    if hasattr(layer, 'weight_mask'):
        mask = mask + layer.weight_mask
        mask[mask>0]=1
        layer.weight_mask = mask
    else:
        layer.register_buffer( 'weight_mask', mask )
    
    layer.register_forward_pre_hook( _mask_weight_hook )
    return layer

def mask_bias(layer, mask, inplace=True):
    """Unstructed pruning for convolution layer

    Args:
        layer: a convolution layer.
        mask: 0-1 mask.
    """
    if not inplace:
        layer = deepcopy(layer)
    if layer.bias is None or mask.shape != layer.bias.shape:
        return layer
    
    mask = torch.tensor( mask, dtype=layer.weight.dtype, device=layer.weight.device, requires_grad=False )
    if hasattr(layer, 'bias_mask'):
        mask = mask + layer.bias_mask
        mask[mask>0]=1
        layer.bias_mask = mask
    else:
        layer.register_buffer( 'bias_mask', mask )
    layer.register_forward_pre_hook( _mask_bias_hook )
    return layer
