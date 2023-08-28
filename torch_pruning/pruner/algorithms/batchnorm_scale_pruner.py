import torch
import torch.nn as nn
import typing 
from .metapruner import MetaPruner
from .scheduler import linear_scheduler
from .. import function
from ... import ops
from ..importance import MagnitudeImportance

class BNScalePruner(MetaPruner):
    """Learning Efficient Convolutional Networks through Network Slimming, 
    https://arxiv.org/abs/1708.06519
    """
    
    def __init__(
        self,
  
        # Basic
        model: nn.Module, # a simple pytorch model
        example_inputs: torch.Tensor, # a dummy input for graph tracing. Should be on the same 
        importance: typing.Callable, # tp.importance.Importance for group importance estimation
        reg=1e-5, # regularization coefficient
        group_lasso=False, # use group lasso
        global_pruning: bool = False, # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        ch_sparsity: float = 0.5,  # channel/dim sparsity, also known as pruning ratio
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None, # layer-specific sparsity, will cover ch_sparsity if specified
        max_ch_sparsity: float = 1.0, # maximum sparsity. useful if over-pruning happens.
        iterative_steps: int = 1,  # for iterative pruning
        iterative_sparsity_scheduler: typing.Callable = linear_scheduler, # scheduler for iterative pruning.
        ignored_layers: typing.List[nn.Module] = None, # ignored layers
        round_to: int = None,  # round channels to the nearest multiple of round_to

        # Advanced
        in_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer input
        out_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer output
        customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None, # pruners for customized layers. E.g., {nn.Linear: my_linear_pruner}
        unwrapped_parameters: typing.Dict[nn.Parameter, int] = None, # unwrapped nn.Parameters & pruning_dims. For example, {ViT.pos_emb: 0}
        root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        forward_fn: typing.Callable = None, # a function to execute model.forward
        output_transform: typing.Callable = None, # a function to transform network outputs

        # deprecated
        channel_groups: typing.Dict[nn.Module, int] = dict(), # channel groups for layers

    ):
        super(BNScalePruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            global_pruning=global_pruning,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict,
            max_ch_sparsity=max_ch_sparsity,
            iterative_steps=iterative_steps,
            iterative_sparsity_scheduler=iterative_sparsity_scheduler,
            ignored_layers=ignored_layers,
            round_to=round_to,
            
            in_channel_groups=in_channel_groups,
            out_channel_groups=out_channel_groups,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            root_module_types=root_module_types,
            forward_fn=forward_fn,
            output_transform=output_transform,
            
            channel_groups=channel_groups,
        )
        self.reg = reg
        self._groups = list(self.DG.get_all_groups())
        self.group_lasso = group_lasso
        if self.group_lasso:
            self._l2_imp = MagnitudeImportance(p=2, group_reduction='mean', normalizer=None, target_types=[nn.modules.batchnorm._BatchNorm])
    
    def regularize(self, model, reg=None):
        if reg is None:
            reg = self.reg # use the default reg

        if self.group_lasso==False:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine==True and m not in self.ignored_layers:
                    if layer.weight.grad is None: continue
                    m.weight.grad.data.add_(reg*torch.sign(m.weight.data))
        else:
            for group in self._groups:
                group_l2norm_sq = self._l2_imp(group)
                if group_l2norm_sq is None:
                    continue
                for dep, _ in group:
                    layer = dep.layer
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm) and layer.affine==True and layer not in self.ignored_layers:
                        if layer.weight.grad is None: continue
                        layer.weight.grad.data.add_(reg * (1 / group_l2norm_sq.sqrt()) * layer.weight.data) # Group Lasso https://tibshirani.su.domains/ftp/sparse-grlasso.pdf