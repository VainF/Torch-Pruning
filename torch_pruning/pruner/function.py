import torch
import torch.nn as nn
from copy import deepcopy
from abc import ABC, abstractclassmethod
from typing import Sequence, Tuple

from .. import ops

__all__=[
    'BasePruningFunc',
    'PrunerBox',

    'prune_conv_out_channels',
    'prune_conv_in_channels',
    'prune_depthwise_conv_out_channels',
    'prune_depthwise_conv_in_channels',
    'prune_batchnorm_out_channels',
    'prune_batchnorm_in_channels',
    'prune_linear_out_channels',
    'prune_linear_in_channels',
    'prune_prelu_out_channels',
    'prune_prelu_in_channels',
    'prune_layernorm_out_channels',
    'prune_layernorm_in_channels',
    'prune_embedding_out_channels',
    'prune_embedding_in_channels',
    'prune_parameter_out_channels',
    'prune_parameter_in_channels',
    'prune_multihead_attention_out_channels',
    'prune_multihead_attention_in_channels',
    'prune_groupnorm_out_channels',
    'prune_groupnorm_in_channels',
    'prune_instancenorm_out_channels',
    'prune_instancenorm_in_channels',
]

class BasePruningFunc(ABC):
    """ Base class for layer pruner.
    It should provide the following functionalities:
        - prune_out_channels: prune out channels of a layer
        - prune_in_channels: prune in channels of a layer
        - get_out_channels: get the number of output channels of a layer
        - get_in_channels: get the number of input channels of a layer
    
    To build the intra-layer dependency, please specify prune_out_channels = prune_in_channels. 

    Example:
    ```python
    class MyPruner(BasePruningFunc):
        def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
            # prune out channels of a layer
            pass
        prune_in_channels = prune_out_channels # this line enables the intra-layer dependency
    ```

    If prune_out_channels != prune_in_channels, there will be no intra-layer dependency.
    """
    TARGET_MODULES = ops.TORCH_OTHERS  # None

    def __init__(self, pruning_dim=1):
        self.pruning_dim = pruning_dim

    @abstractclassmethod
    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]):
        raise NotImplementedError

    @abstractclassmethod
    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]):
        raise NotImplementedError

    @abstractclassmethod
    def get_out_channels(self, layer: nn.Module):
        raise NotImplementedError

    @abstractclassmethod
    def get_in_channels(self, layer: nn.Module):
        raise NotImplementedError

    def check(self, layer, idxs, to_output):
        if self.TARGET_MODULES is not None:
            assert isinstance(layer, self.TARGET_MODULES), 'Mismatched pruner {} and module {}'.format(
                self.__str__, layer)
        if to_output:
            prunable_channels = self.get_out_channels(layer)
        else:
            prunable_channels = self.get_in_channels(layer)
        if prunable_channels is not None:
            assert all(idx < prunable_channels and idx >=
                       0 for idx in idxs), "All pruning indices should fall into [{}, {})".format(0, prunable_channels)

    def __call__(self, layer: nn.Module, idxs: Sequence[int], to_output: bool = True, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
        idxs.sort()
        self.check(layer, idxs, to_output)
        pruning_fn = self.prune_out_channels if to_output else self.prune_in_channels
        if not inplace:
            layer = deepcopy(layer)
        layer = pruning_fn(layer, idxs)
        return layer

    def get_in_channel_groups(self, layer):
        return 1
    
    def get_out_channel_groups(self, layer):
        return 1

    def _prune_parameter_and_grad(self, weight, keep_idxs, pruning_dim):
        pruned_weight = torch.nn.Parameter(torch.index_select(weight, pruning_dim, torch.LongTensor(keep_idxs).to(weight.device).contiguous()))
        if weight.grad is not None:
            pruned_weight.grad = torch.index_select(weight.grad, pruning_dim, torch.LongTensor(keep_idxs).to(weight.device))
        return pruned_weight.to(weight.device)

class ConvPruner(BasePruningFunc):
    TARGET_MODULE = ops.TORCH_CONV

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        keep_idxs.sort()
        layer.out_channels = layer.out_channels-len(idxs)
        if not layer.transposed:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        else:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)
        
        if layer.bias is not None:
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        keep_idxs.sort()
        layer.in_channels = layer.in_channels - len(idxs)
        if layer.groups>1:
            keep_idxs = keep_idxs[:len(keep_idxs)//layer.groups]
        
        if not layer.transposed:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)
        else:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        # no bias pruning because it does not change the output channels
        return layer

    def get_out_channels(self, layer):
        return layer.out_channels

    def get_in_channels(self, layer):
        return layer.in_channels

    def get_in_channel_groups(self, layer):
        return layer.groups
    
    def get_out_channel_groups(self, layer):
        return layer.groups


class DepthwiseConvPruner(ConvPruner):
    TARGET_MODULE = ops.TORCH_CONV

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        keep_idxs.sort()
        layer.out_channels = layer.out_channels-len(idxs)
        layer.in_channels = layer.in_channels-len(idxs)
        layer.groups = layer.groups-len(idxs)
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        if layer.bias is not None:
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        return layer

    prune_in_channels = prune_out_channels
    # def prune_input(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
    #    return self.prune_output(layer, idxs)


class LinearPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_LINEAR

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.out_features)) - set(idxs))
        keep_idxs.sort()
        layer.out_features = layer.out_features-len(idxs)
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        if layer.bias is not None:
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.in_features)) - set(idxs))
        keep_idxs.sort()
        layer.in_features = layer.in_features-len(idxs)
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)
        return layer

    def get_out_channels(self, layer):
        return layer.out_features

    def get_in_channels(self, layer):
        return layer.in_features


class BatchnormPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_BATCHNORM

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.num_features)) - set(idxs))
        keep_idxs.sort()
        layer.num_features = layer.num_features-len(idxs)
        layer.running_mean = layer.running_mean.data[keep_idxs]
        layer.running_var = layer.running_var.data[keep_idxs]

        if layer.affine:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        return layer

    prune_in_channels = prune_out_channels
    # def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
    #    return self.prune_out_channels(layer=layer, idxs=idxs)

    def get_out_channels(self, layer):
        return layer.num_features

    def get_in_channels(self, layer):
        return layer.num_features


class LayernormPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_LAYERNORM

    def __init__(self, metrcis=None, pruning_dim=-1):
        super().__init__(metrcis)
        self.pruning_dim = pruning_dim

    def check(self, layer, idxs):
        layer.dim = self.pruning_dim

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        pruning_dim = self.pruning_dim
        if len(layer.normalized_shape) < -pruning_dim:
            return layer
        num_features = layer.normalized_shape[pruning_dim]
        keep_idxs = torch.tensor(list(set(range(num_features)) - set(idxs)))
        keep_idxs.sort()
        if layer.elementwise_affine:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, pruning_dim)
            if layer.bias is not None:
                layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, pruning_dim)
        if pruning_dim != -1:
            layer.normalized_shape = layer.normalized_shape[:pruning_dim] + (
                keep_idxs.size(0), ) + layer.normalized_shape[pruning_dim+1:]
        else:
            layer.normalized_shape = layer.normalized_shape[:pruning_dim] + (
                keep_idxs.size(0), )
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.normalized_shape[self.pruning_dim]

    def get_in_channels(self, layer):
        return layer.normalized_shape[self.pruning_dim]

class GroupNormPruner(BasePruningFunc):
    def prune_out_channels(self, layer: nn.PReLU, idxs: list) -> nn.Module:
        keep_idxs = list(set(range(layer.num_channels)) - set(idxs))
        keep_idxs.sort()
        layer.num_channels = layer.num_channels-len(idxs)
        if layer.affine:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        return layer
    
    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.num_channels

    def get_in_channels(self, layer):
        return layer.num_channels

    def get_in_channel_groups(self, layer):
        return layer.num_groups
    
    def get_out_channel_groups(self, layer):
        return layer.num_groups

class InstanceNormPruner(BasePruningFunc):
    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.num_features)) - set(idxs))
        keep_idxs.sort()
        layer.num_features = layer.num_features-len(idxs)
        if layer.affine:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.num_features

    def get_in_channels(self, layer):
        return layer.num_features


class PReLUPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_PRELU

    def prune_out_channels(self, layer: nn.PReLU, idxs: list) -> nn.Module:
        if layer.num_parameters == 1:
            return layer
        keep_idxs = list(set(range(layer.num_parameters)) - set(idxs))
        keep_idxs.sort()
        layer.num_parameters = layer.num_parameters-len(idxs)
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        return layer

    prune_in_channels = prune_out_channels

    # def prune_in_channels(self, layer:nn.Module, idxs: Sequence[int]) -> nn.Module:
    #    return self.prune_out_channels(layer=layer, idxs=idxs)

    def get_out_channels(self, layer):
        if layer.num_parameters == 1:
            return None
        else:
            return layer.num_parameters

    def get_in_channels(self, layer):
        return self.get_out_channels(layer=layer)

class EmbeddingPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_EMBED

    def prune_out_channels(self, layer: nn.Embedding, idxs: list) -> nn.Module:
        num_features = layer.embedding_dim
        keep_idxs = list(set(range(num_features)) - set(idxs))
        keep_idxs.sort()
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)
        layer.embedding_dim = len(keep_idxs)
        return layer

    prune_in_channels = prune_out_channels

    # def prune_in_channels(self, layer: nn.Embedding, idxs: list)-> nn.Module:
    #    return self.prune_out_channels(layer=layer, idxs=idxs)

    def get_out_channels(self, layer):
        return layer.embedding_dim

    def get_in_channels(self, layer):
        return self.get_out_channels(layer=layer)

class LSTMPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_LSTM

    def prune_out_channels(self, layer: nn.LSTM, idxs: list) -> nn.Module:
        assert layer.num_layers==1
        num_layers = layer.num_layers
        num_features = layer.hidden_size
        keep_idxs = list(set(range(num_features)) - set(idxs))
        keep_idxs.sort()
        keep_idxs = torch.tensor(keep_idxs)
        expanded_keep_idxs = torch.cat([ keep_idxs+i*num_features for i in range(4) ], dim=0)
        if layer.bidirectional:
            postfix = ['', '_reverse']
        else:
            postfix = ['']
        #for l in range(num_layers):
        for pf in postfix:
            setattr(layer, 'weight_hh_l0'+pf, self._prune_parameter_and_grad(
                getattr(layer, 'weight_hh_l0'+pf), keep_idxs, 0))
            if layer.bias:
                setattr(layer, 'bias_hh_l0'+pf, self._prune_parameter_and_grad(
                    getattr(layer, 'bias_hh_l0'+pf), keep_idxs, 0))
            setattr(layer, 'weight_hh_l0'+pf, self._prune_parameter_and_grad(
                getattr(layer, 'weight_hh_l0'+pf), keep_idxs, 0))
            setattr(layer, 'weight_ih_l0'+pf,  self._prune_parameter_and_grad(
                getattr(layer, 'weight_ih_l0'+pf), expanded_keep_idxs, 1))
            if layer.bias:
                setattr(layer, 'bias_ih_l0'+pf, self._prune_parameter_and_grad(
                    getattr(layer, 'bias_ih_l0'+pf), keep_idxs, 0))
        layer.hidden_size = len(keep_idxs)

    def prune_in_channels(self, layer: nn.LSTM, idxs: list):
        num_features = layer.input_size
        keep_idxs = list(set(range(num_features)) - set(idxs))
        keep_idxs.sort()
        setattr(layer, 'weight_ih_l0', self._prune_parameter_and_grad(
                    getattr(layer, 'weight_ih_l0'), keep_idxs, 1))
        if layer.bidirectional:
            setattr(layer, 'weight_ih_l0_reverse', self._prune_parameter_and_grad(
                    getattr(layer, 'weight_ih_l0_reverse'), keep_idxs, 1))
        layer.input_size = len(keep_idxs)

    def get_out_channels(self, layer):
        return layer.hidden_size
        
    def get_in_channels(self, layer):
        return layer.input_size
    

class ParameterPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_PARAMETER
    def __init__(self, pruning_dim=-1):
        super().__init__(pruning_dim=pruning_dim)
        
    def prune_out_channels(self, tensor, idxs: list) -> nn.Module:
        keep_idxs = list(set(range(tensor.data.shape[self.pruning_dim])) - set(idxs))
        keep_idxs.sort()
        pruned_parameter = self._prune_parameter_and_grad(tensor, keep_idxs, self.pruning_dim)
        return pruned_parameter

    prune_in_channels = prune_out_channels

    def get_out_channels(self, parameter):
        return parameter.shape[self.pruning_dim]

    def get_in_channels(self, parameter):
        return parameter.shape[self.pruning_dim]


class MultiheadAttentionPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_MHA

    def check(self, layer, idxs, to_output):
        super().check(layer, idxs, to_output)
        assert (layer.embed_dim - len(idxs)) % layer.num_heads == 0, "embed_dim (%d) of MultiheadAttention after pruning must divide evenly by `num_heads` (%d)" % (layer.embed_dim, layer.num_heads)

    def prune_out_channels(self, layer, idxs: list) -> nn.Module:
        keep_idxs = list(set(range(layer.embed_dim)) - set(idxs))
        keep_idxs.sort()


        if layer.q_proj_weight is not None:
            layer.q_proj_weight = self._prune_parameter_and_grad(layer.q_proj_weight, keep_idxs, 0)
        if layer.k_proj_weight is not None:
            layer.k_proj_weight = self._prune_parameter_and_grad(layer.k_proj_weight, keep_idxs, 0)
        if layer.v_proj_weight is not None:
            layer.v_proj_weight = self._prune_parameter_and_grad(layer.v_proj_weight, keep_idxs, 0)


        pruning_idxs_repeated = idxs + \
            [i+layer.embed_dim for i in idxs] + \
            [i+2*layer.embed_dim for i in idxs]
        keep_idxs_3x_repeated = list(
            set(range(3*layer.embed_dim)) - set(pruning_idxs_repeated))
        keep_idxs_3x_repeated.sort()
        if layer.in_proj_weight is not None:
            layer.in_proj_weight = self._prune_parameter_and_grad(layer.in_proj_weight, keep_idxs_3x_repeated, 0)
            layer.in_proj_weight = self._prune_parameter_and_grad(layer.in_proj_weight, keep_idxs, 1)
        if layer.in_proj_bias is not None:
            layer.in_proj_bias = self._prune_parameter_and_grad(layer.in_proj_bias, keep_idxs_3x_repeated, 0)

        if layer.bias_k is not None:
            layer.bias_k = self._prune_parameter_and_grad(layer.bias_k, keep_idxs, 2)
        if layer.bias_v is not None:
            layer.bias_v = self._prune_parameter_and_grad(layer.bias_v, keep_idxs, 2)

        linear = layer.out_proj
        keep_idxs = list(set(range(linear.out_features)) - set(idxs))
        keep_idxs.sort()
        linear.out_features = linear.out_features-len(idxs)
        linear.weight = self._prune_parameter_and_grad(linear.weight, keep_idxs, 0)
        if linear.bias is not None:
            linear.bias = self._prune_parameter_and_grad(linear.bias, keep_idxs, 0)
        keep_idxs = list(set(range(linear.in_features)) - set(idxs))
        keep_idxs.sort()
        linear.in_features = linear.in_features-len(idxs)
        linear.weight = self._prune_parameter_and_grad(linear.weight, keep_idxs, 1)
        layer.embed_dim = layer.embed_dim - len(idxs)
        layer.head_dim = layer.embed_dim // layer.num_heads
        layer.kdim = layer.embed_dim
        layer.vdim = layer.embed_dim
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.embed_dim

    def get_in_channels(self, layer):
        return self.get_out_channels(layer)

PrunerBox = {
    ops.OPTYPE.CONV: ConvPruner(),
    ops.OPTYPE.LINEAR: LinearPruner(),
    ops.OPTYPE.BN: BatchnormPruner(),
    ops.OPTYPE.DEPTHWISE_CONV: DepthwiseConvPruner(),
    ops.OPTYPE.PRELU: PReLUPruner(),
    ops.OPTYPE.LN: LayernormPruner(),
    ops.OPTYPE.EMBED: EmbeddingPruner(),
    ops.OPTYPE.PARAMETER: ParameterPruner(),
    ops.OPTYPE.MHA: MultiheadAttentionPruner(),
    ops.OPTYPE.LSTM: LSTMPruner(),
    ops.OPTYPE.GN: GroupNormPruner(),
    ops.OPTYPE.IN: InstanceNormPruner(),
}

# Alias
prune_conv_out_channels = PrunerBox[ops.OPTYPE.CONV].prune_out_channels
prune_conv_in_channels = PrunerBox[ops.OPTYPE.CONV].prune_in_channels

prune_depthwise_conv_out_channels = PrunerBox[ops.OPTYPE.DEPTHWISE_CONV].prune_out_channels
prune_depthwise_conv_in_channels = PrunerBox[ops.OPTYPE.DEPTHWISE_CONV].prune_in_channels

prune_batchnorm_out_channels = PrunerBox[ops.OPTYPE.BN].prune_out_channels
prune_batchnorm_in_channels = PrunerBox[ops.OPTYPE.BN].prune_in_channels

prune_linear_out_channels = PrunerBox[ops.OPTYPE.LINEAR].prune_out_channels
prune_linear_in_channels = PrunerBox[ops.OPTYPE.LINEAR].prune_in_channels

prune_prelu_out_channels = PrunerBox[ops.OPTYPE.PRELU].prune_out_channels
prune_prelu_in_channels = PrunerBox[ops.OPTYPE.PRELU].prune_in_channels

prune_layernorm_out_channels = PrunerBox[ops.OPTYPE.LN].prune_out_channels
prune_layernorm_in_channels = PrunerBox[ops.OPTYPE.LN].prune_in_channels

prune_embedding_out_channels = PrunerBox[ops.OPTYPE.EMBED].prune_out_channels
prune_embedding_in_channels = PrunerBox[ops.OPTYPE.EMBED].prune_in_channels

prune_parameter_out_channels = PrunerBox[ops.OPTYPE.PARAMETER].prune_out_channels
prune_parameter_in_channels = PrunerBox[ops.OPTYPE.PARAMETER].prune_in_channels

prune_multihead_attention_out_channels = PrunerBox[ops.OPTYPE.MHA].prune_out_channels
prune_multihead_attention_in_channels = PrunerBox[ops.OPTYPE.MHA].prune_in_channels

prune_lstm_out_channels = PrunerBox[ops.OPTYPE.LSTM].prune_out_channels
prune_lstm_in_channels = PrunerBox[ops.OPTYPE.LSTM].prune_in_channels

prune_groupnorm_out_channels = PrunerBox[ops.OPTYPE.GN].prune_out_channels
prune_groupnorm_in_channels = PrunerBox[ops.OPTYPE.GN].prune_in_channels

prune_instancenorm_out_channels = PrunerBox[ops.OPTYPE.IN].prune_out_channels
prune_instancenorm_in_channels = PrunerBox[ops.OPTYPE.IN].prune_in_channels
