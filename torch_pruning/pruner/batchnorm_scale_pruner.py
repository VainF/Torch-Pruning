from .. import dependency, functional, utils
from numbers import Number
from typing import Callable
from .basepruner import MetaPruner
from .scheduler import linear_scheduler
import torch
import torch.nn as nn

class BNScalePruner(MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        reg=1e-5,
        pruning_steps=1,
        pruning_rate_scheduler: Callable = linear_scheduler,
        ch_sparsity=0.5,
        layer_ch_sparsity=None,
        global_pruning=False,
        max_ch_sparsity=1.0,
        round_to=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
    ):
        super(BNScalePruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            pruning_steps=pruning_steps,
            pruning_rate_scheduler=pruning_rate_scheduler,
            ch_sparsity=ch_sparsity,
            layer_ch_sparsity=layer_ch_sparsity,
            global_pruning=global_pruning,
            max_ch_sparsity=max_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            user_defined_parameters=user_defined_parameters,
            output_transform=output_transform,
        )
        self.reg = reg

    def regularize(self, model, loss):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine==True:
                m.weight.grad.data.add_(self.reg*torch.sign(m.weight.data))
