from numpy import isin
import torch
import torch.nn as nn
from copy import deepcopy
from functools import reduce
from operator import mul
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Callable, Sequence, Tuple, Dict

class BasePruner(ABC):
    def __init__(self, metrics_dict: Dict[str, Callable]=None, dim=1):
        self.metrics = None
        self.set_metrics(metrics_dict)
        self.dim=dim
    
    def add_metric(self, name, metric_fn):
        self.metrics[name] = metric_fn

    def set_metrics(self, metric_dict=None):
        if metric_dict is None:
            metric_dict = {"#params": self.calc_nparams_to_prune}
        self.metrics = metric_dict

    def check(self, layer, idxs):
        pass

    def __call__(self, layer: nn.Module, idxs: Sequence[int], inplace: bool=True, dry_run: bool=False) -> Tuple[nn.Module, int]:
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
        keep_idxs.sort()
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
        keep_idxs.sort()
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
        keep_idxs.sort()
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
        keep_idxs.sort()
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
        keep_idxs.sort()
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
        keep_idxs.sort()
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
        keep_idxs.sort()
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
        keep_idxs.sort()
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        layer.embedding_dim = len(keep_idxs)
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Embedding, idxs: Sequence[int]) -> int:
        nparams_to_prune = layer.num_embeddings * len(idxs)
        return nparams_to_prune

class ParameterPruner(BasePruner):
    def prune(self, tensor, idxs: list)-> nn.Module:
        keep_idxs = list(set(range(tensor.data.shape[self.dim])) - set(idxs))
        keep_idxs.sort()
        tensor.data = torch.index_select(tensor.data, self.dim, torch.LongTensor(keep_idxs))
        return tensor

    @staticmethod
    def calc_nparams_to_prune(tensor: nn.Embedding, idxs: Sequence[int]) -> int:
        return 0
    
class MultiheadAttentionPruner(BasePruner):
    def check(self, layer, idxs):
        assert (layer.embed_dim - len(idxs))  % layer.num_heads == 0, "embed_dim (%d) of MultiheadAttention after pruning must divide evenly by `num_heads` (%d)"%(layer.embed_dim, layer.num_heads)
    def prune(self, layer, idxs: list)-> nn.Module:
        keep_idxs = list(set(range(layer.embed_dim)) - set(idxs))
        keep_idxs.sort()
        if layer.q_proj_weight is not None:
            layer.q_proj_weight.data = torch.index_select(layer.q_proj_weight.data, 0, torch.LongTensor(keep_idxs))
        if layer.k_proj_weight is not None:
            layer.q_proj_weight.data = torch.index_select(layer.q_proj_weight.data, 0, torch.LongTensor(keep_idxs))
        if layer.v_proj_weight is not None:
            layer.v_proj_weight.data = torch.index_select(layer.v_proj_weight.data, 0, torch.LongTensor(keep_idxs))
        
        pruning_idxs_3x = idxs + [i+layer.embed_dim for i in idxs] + [i+2*layer.embed_dim for i in idxs]
        keep_idxs_3x = list(set(range(3*layer.embed_dim)) - set(pruning_idxs_3x))
        keep_idxs_3x.sort()
        if layer.in_proj_weight is not None:
            layer.in_proj_weight.data = torch.index_select(layer.in_proj_weight.data, 0, torch.LongTensor(keep_idxs_3x))
            layer.in_proj_weight.data = torch.index_select(layer.in_proj_weight.data, 1, torch.LongTensor(keep_idxs))
            
        if layer.in_proj_bias is not None:
            layer.in_proj_bias.data = torch.index_select(layer.in_proj_bias.data, 0, torch.LongTensor(keep_idxs_3x))
        
        if layer.bias_k is not None:
            layer.bias_k.data = torch.index_select(layer.bias_k.data, 2, torch.LongTensor(keep_idxs))
        if layer.bias_v is not None:
            layer.bias_v.data = torch.index_select(layer.bias_v.data, 2, torch.LongTensor(keep_idxs))
        
        linear = layer.out_proj
        keep_idxs = list(set(range(linear.out_features)) - set(idxs))
        keep_idxs.sort()
        linear.out_features = linear.out_features-len(idxs)
        linear.weight = torch.nn.Parameter(linear.weight.data.clone()[keep_idxs])
        if linear.bias is not None:
            linear.bias = torch.nn.Parameter(linear.bias.data.clone()[keep_idxs])
        keep_idxs = list(set(range(linear.in_features)) - set(idxs))
        keep_idxs.sort()
        linear.in_features = linear.in_features-len(idxs)
        linear.weight = torch.nn.Parameter(linear.weight.data.clone()[:, keep_idxs])
        layer.embed_dim = layer.embed_dim - len(idxs)
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Embedding, idxs: Sequence[int]) -> int:
        linear = layer.out_proj
        nparams_to_prune = len(idxs)*linear.weight.shape[1] + len(idxs) * (layer.embed_dim - len(idxs)) + (len(idxs) if linear.bias is not None else 0) 
        return nparams_to_prune


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
prune_multihead_attention = MultiheadAttentionPruner()

def set_global_metrics(metric_dict):
    prune_conv_in_channel.set_metrics(metric_dict)
    prune_conv_out_channel.set_metrics(metric_dict)
    prune_group_conv.set_metrics(metric_dict)
    prune_batchnorm.set_metrics(metric_dict)
    prune_linear_in_channel.set_metrics(metric_dict)
    prune_linear_out_channel.set_metrics(metric_dict)
    prune_prelu.set_metrics(metric_dict)
    prune_layernorm.set_metrics(metric_dict)
    prune_embedding.set_metrics(metric_dict)
    prune_parameter.set_metrics(metric_dict)
    prune_multihead_attention.set_metrics(metric_dict)
