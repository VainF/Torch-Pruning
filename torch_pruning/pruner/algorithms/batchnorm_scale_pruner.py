from numbers import Number
from typing import Callable
from .metapruner import MetaPruner
from .scheduler import linear_scheduler
import torch
import torch.nn as nn
import math

from ..importance import MagnitudeImportance

class BNScalePruner(MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        reg=1e-5,
        group_lasso=False,
        iterative_steps=1,
        iterative_sparsity_scheduler: Callable = linear_scheduler,
        ch_sparsity=0.5,
        ch_sparsity_dict=None,
        global_pruning=False,
        max_ch_sparsity=1.0,
        round_to=None,
        ignored_layers=None,
        customized_pruners=None,
        unwrapped_parameters=None,
        output_transform=None,
    ):
        super(BNScalePruner, self).__init__(
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
        self.reg = reg
        self._groups = list(self.DG.get_all_groups())
        self.group_lasso = True
        if self.group_lasso:
            self._l2_imp = MagnitudeImportance(p=2, group_reduction='mean', normalizer=None, target_types=[nn.modules.batchnorm._BatchNorm])
    
    def regularize(self, model, reg=None):
        if reg is None:
            reg = self.reg # use the default reg

        if self.group_lasso==False:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine==True and m not in self.ignored_layers:
                    m.weight.grad.data.add_(reg*torch.sign(m.weight.data))
        else:
            for group in self._groups:
                group_l2norm_sq, group_size = self._l2_imp(group, return_group_size=True)
                if group_l2norm_sq is None:
                    continue
                for dep, _ in group:
                    layer = dep.layer
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm) and layer.affine==True and layer not in self.ignored_layers:
                        layer.weight.grad.data.add_(reg * math.sqrt(group_size) * (1 / group_l2norm_sq.sqrt()) * layer.weight.data) # Group Lasso https://tibshirani.su.domains/ftp/sparse-grlasso.pdf