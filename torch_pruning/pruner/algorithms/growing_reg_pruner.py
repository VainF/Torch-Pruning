from .metapruner import MetaPruner
from .scheduler import linear_scheduler
import typing
import torch
import torch.nn as nn

from ..importance import MagnitudeImportance, GroupNormImportance
from .. import function
import math

class GrowingRegPruner(MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        reg=1e-5,
        delta_reg = 1e-5,
        iterative_steps=1,
        iterative_sparsity_scheduler: typing.Callable = linear_scheduler,
        ch_sparsity=0.5,
        ch_sparsity_dict=None,
        global_pruning=False,
        max_ch_sparsity=1.0,
        round_to=None,
        ignored_layers=None,
        customized_pruners=None,
        unwrapped_parameters=None,
        output_transform=None,
        target_types=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm],
    ):
        super(GrowingRegPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            iterative_steps=iterative_steps,
            iterative_sparsity_scheduler=iterative_sparsity_scheduler,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict,
            global_pruning=global_pruning,
            max_ch_sparsity=max_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            output_transform=output_transform,
        )
        self.base_reg = reg
        self._groups = list(self.DG.get_all_groups())
        self.group_lasso = True
        self._l2_imp = GroupNormImportance()

        group_reg = {}
        for group in self._groups:
            group_reg[group] = torch.ones( len(group[0].idxs) ) * self.base_reg
        self.group_reg = group_reg
        self.delta_reg = delta_reg

    def update_reg(self):
        for group in self._groups:
            group_l2norm_sq = self._l2_imp(group)
            if group_l2norm_sq is None:
                continue 
            reg = self.group_reg[group]
            standarized_imp = (group_l2norm_sq.max() - group_l2norm_sq) / (group_l2norm_sq.max() - group_l2norm_sq.min() + 1e-8)
            reg = reg + self.delta_reg * standarized_imp.to(reg.device)
            self.group_reg[group] = reg

    def regularize(self, model):
        for i, group in enumerate(self._groups):
            group_l2norm_sq = self._l2_imp(group)
            if group_l2norm_sq is None:
                continue
            
            reg = self.group_reg[group]
            for dep, idxs in group:
                layer = dep.layer
                pruning_fn = dep.pruning_fn
                if isinstance(layer, nn.modules.batchnorm._BatchNorm) and layer.affine==True and layer not in self.ignored_layers:
                    layer.weight.grad.data.add_(reg.to(layer.weight.device) * layer.weight.data)
                elif isinstance(layer, (nn.modules.conv._ConvNd, nn.Linear)) and layer not in self.ignored_layers:
                    if pruning_fn in [function.prune_conv_out_channels, function.prune_linear_out_channels]:
                        w = layer.weight.data[idxs]
                        g = w * reg.to(layer.weight.device).view( -1, *([1]*(len(w.shape)-1)) ) #/ group_norm.view( -1, *([1]*(len(w.shape)-1)) ) * group_size #group_size #* scale.view( -1, *([1]*(len(w.shape)-1)) )
                        layer.weight.grad.data[idxs]+= g
                    elif pruning_fn in [function.prune_conv_in_channels, function.prune_linear_in_channels]:
                        w = layer.weight.data[:, idxs]
                        g = w * reg.to(layer.weight.device).view( 1, -1, *([1]*(len(w.shape)-2))  ) #/ gn.view( 1, -1, *([1]*(len(w.shape)-2)) ) * group_size #* scale.view( 1, -1, *([1]*(len(w.shape)-2))  )
                        layer.weight.grad.data[:, idxs]+=g