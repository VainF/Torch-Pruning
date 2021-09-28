import torch
import torch.nn as nn
from copy import deepcopy
from functools import reduce
from operator import mul
from abc import ABC, abstractstaticmethod
from typing import Sequence, Tuple

#__all__=['prune_conv', 'prune_related_conv', 'prune_linear', 'prune_related_linear', 'prune_batchnorm', 'prune_prelu', 'prune_group_conv']

# Class
class BasePruningFunction(ABC):
    """Base pruning function
    """

    NEED_PRUNING_DIM_MODULES = [ nn.modules.normalization.LayerNorm, ]

    @classmethod
    def apply(cls, layer: nn.Module, idxs: Sequence[int], inplace: bool=True, dry_run: bool=False, **kwargs) -> Tuple[nn.Module, int] : 
        idxs = list(set(idxs))
        cls.check(layer, idxs)
        if any(isinstance(layer, submodule) for submodule in cls.NEED_PRUNING_DIM_MODULES):
            nparams_to_prune = cls.calc_nparams_to_prune(layer, idxs, pruning_dim=kwargs['pruning_dim'])
        else:
            nparams_to_prune = cls.calc_nparams_to_prune(layer, idxs)
        if dry_run:
            return layer, nparams_to_prune
        if not inplace:
            layer = deepcopy(layer)
        if any(isinstance(layer, submodule) for submodule in cls.NEED_PRUNING_DIM_MODULES):
            layer = cls.prune_params(layer, idxs, pruning_dim=kwargs['pruning_dim'])
        else:
            layer = cls.prune_params(layer, idxs)
        return layer, nparams_to_prune

    @staticmethod
    def check(layer: nn.Module, idxs: Sequence[int]) -> None: 
        pass

    @abstractstaticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        pass

    @abstractstaticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int: 
        pass

class ConvPruning(BasePruningFunction):

    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module: 
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        layer.out_channels = layer.out_channels-len(idxs)
        if not layer.transposed:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        else:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer
    
    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int: 
        nparams_to_prune = len(idxs) * reduce(mul, layer.weight.shape[1:]) + (len(idxs) if layer.bias is not None else 0)
        return nparams_to_prune


class GroupConvPruning(ConvPruning):

    @staticmethod
    def check(layer, idxs)-> nn.Module: 
        if layer.groups>1:
            assert layer.groups==layer.in_channels and layer.groups==layer.out_channels, "only group conv with in_channel==groups==out_channels is supported"
    
    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module: 
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        layer.out_channels = layer.out_channels-len(idxs)
        layer.in_channels = layer.in_channels-len(idxs)
        layer.groups = layer.groups-len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer

class RelatedConvPruning(BasePruningFunction):
    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module: 
        keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        layer.in_channels = layer.in_channels - len(idxs)
        if not layer.transposed:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs]) 
        else:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        # no bias pruning because it does not change the output size
        return layer
    
    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int: 
        nparams_to_prune = len(idxs) *  layer.weight.shape[0] * reduce(mul ,layer.weight.shape[2:])
        return nparams_to_prune

class LinearPruning(BasePruningFunction):
    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module: 
        keep_idxs = list(set(range(layer.out_features)) - set(idxs))
        layer.out_features = layer.out_features-len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer
    
    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int: 
        nparams_to_prune = len(idxs)*layer.weight.shape[1] + (len(idxs) if layer.bias is not None else 0)
        return nparams_to_prune

class RelatedLinearPruning(BasePruningFunction):
    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module: 
        keep_idxs = list(set(range(layer.in_features)) - set(idxs))
        layer.in_features = layer.in_features-len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        return layer
    
    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int: 
        nparams_to_prune = len(idxs) *  layer.weight.shape[0]
        return nparams_to_prune
    
class BatchnormPruning(BasePruningFunction):
    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module: 
        keep_idxs = list(set(range(layer.num_features)) - set(idxs))
        layer.num_features = layer.num_features-len(idxs)
        layer.running_mean = layer.running_mean.data.clone()[keep_idxs]
        layer.running_var = layer.running_var.data.clone()[keep_idxs]
        if layer.affine:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer
    
    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int: 
        nparams_to_prune = len(idxs)* ( 2 if layer.affine else 1)
        return nparams_to_prune

class LayernormPruning(BasePruningFunction):
    @staticmethod
    def prune_params(layer:nn.Module, idxs: Sequence[int], pruning_dim: int) -> nn.Module:
        if len(layer.normalized_shape) < -pruning_dim:
            return layer

        num_features = layer.normalized_shape[pruning_dim]
        keep_idxs = torch.tensor(list(set(range(num_features)) - set(idxs)))
        if layer.elementwise_affine:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone().index_select(pruning_dim, keep_idxs))
            layer.bias = torch.nn.Parameter(layer.bias.data.clone().index_select(pruning_dim, keep_idxs))
        
        if pruning_dim != -1:
            layer.normalized_shape = layer.normalized_shape[:pruning_dim] + (keep_idxs.size(0), ) + layer.normalized_shape[pruning_dim+1:]  
        else:
            layer.normalized_shape = layer.normalized_shape[:pruning_dim] + (keep_idxs.size(0), ) 
       
        return layer
    
    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int], pruning_dim: int) -> int:
        nparams_to_prune = len(idxs) * 2 if layer.elementwise_affine and len(layer.normalized_shape) >= -pruning_dim else 0
        return nparams_to_prune

class PReLUPruning(BasePruningFunction):
    @staticmethod
    def prune_params(layer: nn.PReLU, idxs: list)-> nn.Module: 
        if layer.num_parameters==1: return layer
        keep_idxs = list(set(range(layer.num_parameters)) - set(idxs))
        layer.num_parameters = layer.num_parameters-len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        return layer
    
    @staticmethod
    def calc_nparams_to_prune(layer: nn.PReLU, idxs: Sequence[int]) -> int: 
        nparams_to_prune = 0 if layer.num_parameters==1 else len(idxs)
        return nparams_to_prune

class EmbeddingPruning(BasePruningFunction):  
    @staticmethod
    def prune_params(layer: nn.Embedding, idxs: list)-> nn.Module:
        num_features = layer.embedding_dim
        keep_idxs = list(set(range(num_features)) - set(idxs))
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        layer.embedding_dim = len(keep_idxs)
        return layer
    
    @staticmethod
    def calc_nparams_to_prune(layer: nn.Embedding, idxs: Sequence[int]) -> int:
        nparams_to_prune = layer.num_embeddings * len(idxs)
        return nparams_to_prune
        

# Funtional
def prune_conv(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool=True, dry_run: bool=False, **kwargs) -> Tuple[nn.Module, int] : 
    return ConvPruning.apply(layer, idxs, inplace, dry_run, **kwargs)

def prune_related_conv(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool=True, dry_run: bool=False, **kwargs) -> Tuple[nn.Module, int] : 
    return RelatedConvPruning.apply(layer, idxs, inplace, dry_run, **kwargs)

def prune_group_conv(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool=True, dry_run: bool=False, **kwargs) -> Tuple[nn.Module, int] : 
    return GroupConvPruning.apply(layer, idxs, inplace, dry_run, **kwargs)

def prune_batchnorm(layer: nn.modules.batchnorm._BatchNorm, idxs: list, inplace: bool=True, dry_run: bool=False, **kwargs) -> Tuple[nn.Module, int] : 
    return BatchnormPruning.apply(layer, idxs, inplace, dry_run, **kwargs)

def prune_linear(layer: nn.Linear, idxs: list, inplace: bool=True, dry_run: bool=False, **kwargs) -> Tuple[nn.Module, int] : 
    return LinearPruning.apply(layer, idxs, inplace, dry_run, **kwargs)

def prune_related_linear(layer: nn.Linear, idxs: list, inplace: bool=True, dry_run: bool=False, **kwargs) -> Tuple[nn.Module, int] : 
    return RelatedLinearPruning.apply(layer, idxs, inplace, dry_run, **kwargs)

def prune_prelu(layer: nn.PReLU, idxs: list, inplace: bool=True, dry_run: bool=False, **kwargs) -> Tuple[nn.Module, int] : 
    return PReLUPruning.apply(layer, idxs, inplace, dry_run, **kwargs)

def prune_layernorm(layer: nn.modules.normalization.LayerNorm, idxs: list, inplace: bool=True, dry_run: bool=False, **kwargs) -> Tuple[nn.Module, int] :
    return LayernormPruning.apply(layer, idxs, inplace, dry_run, **kwargs)

def prune_embedding(layer: nn.Embedding, idxs: list, inplace: bool=True, dry_run: bool=False, **kwargs) -> Tuple[nn.Module, int] :
    return EmbeddingPruning.apply(layer, idxs, inplace, dry_run, **kwargs)
