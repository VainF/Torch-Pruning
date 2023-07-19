from .metapruner import MetaPruner
from .scheduler import linear_scheduler
import typing
import torch
import torch.nn as nn

class GrowingRegPruner(MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        reg=1e-5,
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
        self.reg = reg

    def regularize(self, model, reg=None):
        if reg is None:
            reg = self.reg # use the default reg
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine==True and m not in self.ignored_layers:
                m.weight.grad.data.add_(reg*torch.sign(m.weight.data))

    