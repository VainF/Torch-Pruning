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
        pruning_ratio: float = 0.5,  # channel/dim pruning ratio
        pruning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific pruning ratios, will cover pruning_ratio if specified
        max_pruning_ratio: float = 1.0, # the maximum pruning ratio. useful if over-pruning happens.
        iterative_steps: int = 1,  # for iterative pruning
        iterative_pruning_ratio_scheduler: typing.Callable = linear_scheduler, # scheduler for iterative pruning.
        ignored_layers: typing.List[nn.Module] = None, # ignored layers
        round_to: int = None,  # round channels to the nearest multiple of round_to

        # Advanced
        in_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer input
        out_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer output
        num_heads: typing.Dict[nn.Module, int] = dict(), # The number of heads for multi-head attention
        prune_num_heads: bool = False, # remove entire heads in multi-head attention
        prune_head_dims: bool = True, # remove head dimensions in multi-head attention
        head_pruning_ratio: float = 0.0, # head pruning ratio
        head_pruning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific head pruning ratio
        customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None, # pruners for customized layers. E.g., {nn.Linear: my_linear_pruner}
        unwrapped_parameters: typing.Dict[nn.Parameter, int] = None, # unwrapped nn.Parameters & pruning_dims. For example, {ViT.pos_emb: 0}
        root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        forward_fn: typing.Callable = None, # a function to execute model.forward
        output_transform: typing.Callable = None, # a function to transform network outputs

        # deprecated
        channel_groups: typing.Dict[nn.Module, int] = dict(), # channel groups for layers
        ch_sparsity: float = None,
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None,
    ):
        super(BNScalePruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            global_pruning=global_pruning,
            pruning_ratio=pruning_ratio,
            pruning_ratio_dict=pruning_ratio_dict,
            max_pruning_ratio=max_pruning_ratio,
            iterative_steps=iterative_steps,
            iterative_pruning_ratio_scheduler=iterative_pruning_ratio_scheduler,
            ignored_layers=ignored_layers,
            round_to=round_to,
            
            in_channel_groups=in_channel_groups,
            out_channel_groups=out_channel_groups,
            num_heads=num_heads,
            prune_num_heads=prune_num_heads,
            prune_head_dims=prune_head_dims,
            head_pruning_ratio=head_pruning_ratio,
            head_pruning_ratio_dict=head_pruning_ratio_dict,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            root_module_types=root_module_types,
            forward_fn=forward_fn,
            output_transform=output_transform,
            
            channel_groups=channel_groups,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict
        )
        self.reg = reg
        self._groups = list(self.DG.get_all_groups(root_module_types=self.root_module_types, ignored_layers=self.ignored_layers))
        self.group_lasso = group_lasso
        if self.group_lasso:
            self._l2_imp = MagnitudeImportance(p=2, group_reduction='mean', normalizer=None, target_types=[nn.modules.batchnorm._BatchNorm])
    
    def update_regularizor(self):
        self._groups = list(self.DG.get_all_groups(root_module_types=self.root_module_types, ignored_layers=self.ignored_layers))

    def regularize(self, model, reg=None, bias=False):
        if reg is None:
            reg = self.reg # use the default reg

        if self.group_lasso==False:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine==True and m not in self.ignored_layers:
                    if m.weight.grad is None: continue
                    m.weight.grad.data.add_(reg*torch.sign(m.weight.data))
        else:
            for group in self._groups:
                group_l2norm_sq = self._l2_imp(group) + 1e-9 # + 1e-9 to avoid inf
                if group_l2norm_sq is None or torch.any(torch.isnan(group_l2norm_sq)):  # avoid nan
                    continue
                gamma = reg * (1 / group_l2norm_sq.sqrt())

                for i, (dep, _) in enumerate(group):
                    layer = dep.layer
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm) and layer.affine==True and layer not in self.ignored_layers:
                        if layer.weight.grad is None: continue
                        root_idxs = group[i].root_idxs
                        _gamma = torch.index_select(gamma, 0, torch.tensor(root_idxs, device=gamma.device))
                        layer.weight.grad.data.add_(_gamma * layer.weight.data) # Group Lasso https://tibshirani.su.domains/ftp/sparse-grlasso.pdf