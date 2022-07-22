from .. import dependency, functional, utils
from numbers import Number
from typing import Callable
from .basepruner import MetaPruner, linear_scheduler
import torch
import torch.nn as nn

class BNScalePruner(MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        reg=1e-5,
        total_steps=1,
        pruning_rate_scheduler: Callable = linear_scheduler,
        ch_sparsity=0.5,
        layer_ch_sparsity=None,
        global_pruning=False,
        global_max_ch_sparsity=1.0,
        round_to=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
    ):
        super(BNScalePruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            total_steps=total_steps,
            pruning_rate_scheduler=pruning_rate_scheduler,
            ch_sparsity=ch_sparsity,
            layer_ch_sparsity=layer_ch_sparsity,
            global_pruning=global_pruning,
            global_max_ch_sparsity=global_max_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            user_defined_parameters=user_defined_parameters,
            output_transform=output_transform,
        )
        self.reg = reg

    def regularize(self, model, loss):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(self.reg*torch.sign(m.weight.data))
