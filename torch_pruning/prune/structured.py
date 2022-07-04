from numpy import isin
import torch
import torch.nn as nn
from copy import deepcopy
from functools import reduce
from operator import mul
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Callable, Sequence, Tuple, Dict

class BasePruner(ABC):
    def __init__(self, metrics: Dict[str, Callable]=None, dim=1):
        if metrics == None:
            metrics = {"#params": self.calc_nparams_to_prune}
        self.metrics = metrics
        self.dim=dim
    
    def add_metric(self, name, metric_fn):
        self.metrics[name] = metric_fn

    def check(self, layer, idxs):
        pass

    def __call__(self, layer: nn.Module, idxs: Sequence[int], inplace: bool=True, dry_run: bool=False) -> Tuple[nn.Module, int]:
        idxs = list(set(idxs))
        self.check(layer, idxs)
        metrics = { name: metric_fn(layer, idxs) for (name, metric_fn) in self.metrics.items() }
        if dry_run:
            return layer, metrics
        if not inplace:
            layer = deepcopy(layer)
        layer = self.prune(layer, idxs)
        return layer, metrics

    @abstractmethod
    def prune(self, layer:nn.Module, idxs:Sequence[int])->nn.Module:
        pass

    @abstractstaticmethod
    def calc_nparams_to_prune(layer, idxs):
        return 0

class ConvOutChannelPruner(BasePruner):
    def prune(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module: 
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


class ConvInChannelPruner(BasePruner):
    def prune(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module: 
        keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        layer.in_channels = layer.in_channels - len(idxs)
        if not layer.transposed:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs]) 
        else:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        # no bias pruning because it does not change the output channels
        return layer
        
    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int: 
        nparams_to_prune = len(idxs) *  layer.weight.shape[0] * reduce(mul ,layer.weight.shape[2:])
        return nparams_to_prune

class GroupConvPruner(ConvOutChannelPruner):

    def check(self, layer, idxs)-> nn.Module: 
        pass

    def prune(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module: 
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        layer.out_channels = layer.out_channels-len(idxs)
        layer.in_channels = layer.in_channels-len(idxs)
        layer.groups = layer.groups-len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer


class LinearOutChannelPruner(BasePruner):
    def prune(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module: 
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

class LinearInChannelPruner(BasePruner):
    def prune(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module: 
        keep_idxs = list(set(range(layer.in_features)) - set(idxs))
        layer.in_features = layer.in_features-len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int: 
        nparams_to_prune = len(idxs) *  layer.weight.shape[0]
        return nparams_to_prune


class BatchnormPruner(BasePruner):
    def prune(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module: 
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


class LayernormPruner(BasePruner):
    def __init__(self, metrcis=None, pruning_dim=-1):
        super().__init__(metrcis)
        self.pruning_dim = pruning_dim

    def check(self, layer, idxs):
        layer.pruning_dim = self.pruning_dim
        
    def prune(self, layer:nn.Module, idxs: Sequence[int]) -> nn.Module:
        pruning_dim = self.pruning_dim
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
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        nparams_to_prune = len(idxs) * 2 if layer.elementwise_affine and len(layer.normalized_shape) >= -layer.pruning_dim else 0
        return nparams_to_prune

class PReLUPruner(BasePruner):
    def prune(self, layer: nn.PReLU, idxs: list)-> nn.Module: 
        if layer.num_parameters==1: return layer
        keep_idxs = list(set(range(layer.num_parameters)) - set(idxs))
        layer.num_parameters = layer.num_parameters-len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        return layer
    
    @staticmethod
    def calc_nparams_to_prune(layer: nn.PReLU, idxs: Sequence[int]) -> int: 
        nparams_to_prune = 0 if layer.num_parameters==1 else len(idxs)
        return nparams_to_prune

class EmbeddingPruner(BasePruner):  
    def prune(self, layer: nn.Embedding, idxs: list)-> nn.Module:
        num_features = layer.embedding_dim
        keep_idxs = list(set(range(num_features)) - set(idxs))
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        layer.embedding_dim = len(keep_idxs)
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Embedding, idxs: Sequence[int]) -> int:
        nparams_to_prune = layer.num_embeddings * len(idxs)
        return nparams_to_prune

class ParameterPruner(BasePruner):
    def prune(self, tensor, idxs: list)-> nn.Module:
        #print("aha", idxs, self.dim, tensor.data.shape)
        keep_idxs = list(set(range(tensor.data.shape[self.dim])) - set(idxs))
        tensor.data = torch.index_select(tensor.data, self.dim, torch.LongTensor(keep_idxs))
        #print(tensor.data.shape)
        return tensor

    @staticmethod
    def calc_nparams_to_prune(tensor: nn.Embedding, idxs: Sequence[int]) -> int:
        return 0
    
prune_conv_in_channel = ConvInChannelPruner()
prune_conv_out_channel = ConvOutChannelPruner()
prune_group_conv = GroupConvPruner()
prune_batchnorm = BatchnormPruner()
prune_linear_in_channel = LinearInChannelPruner()
prune_linear_out_channel = LinearOutChannelPruner()
prune_prelu = PReLUPruner()
prune_layernorm = LayernormPruner()
prune_embedding = EmbeddingPruner() 
prune_parameter = ParameterPruner(dim=2) # default=2 for tranformers



