from numbers import Number
from typing import Callable
from .metapruner import MetaPruner
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

    def regularize(self, model):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine==True:
                m.weight.grad.data.add_(self.reg*torch.sign(m.weight.data))
