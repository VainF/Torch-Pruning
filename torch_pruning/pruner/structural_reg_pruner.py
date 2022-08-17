from .. import dependency, functional, utils
from numbers import Number
from typing import Callable
from .basepruner import MetaPruner
import torch
import torch.nn as nn

class StructrualRegularizedPruner(MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        pruning_steps=1,
        beta=1e-4,
        pruning_rate_scheduler: Callable = None,
        ch_sparsity=0.5,
        layer_ch_sparsity=None,
        global_pruning=False,
        max_ch_sparsity=1.0,
        round_to=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
    ):
        super(StructrualRegularizedPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
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
        self.importance = importance
        self.dropout_groups = {}
        self.beta = beta
        self.plans = self.get_all_cliques()
    
    def estimate_importance(self, clique):
        return self.importance(clique)

    def structrual_dropout(self, module, input, output):
        return self.dropout_groups[module][0](output)

    def regularize(self, model, loss):

        for clique in self.plans:
            for dep, idxs in clique:
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn in [
                    functional.prune_conv_out_channel,
                    functional.prune_linear_out_channel,
                ]:
                    # regularize output channels
                    layer.weight.grad.data.add_(self.beta*torch.sign(layer.weight.data))
                elif prune_fn in [
                    functional.prune_conv_in_channel,
                    functional.prune_linear_in_channel,
                ]:
                    # regularize input channels
                    layer.weight.grad.data.add_(self.beta*torch.sign(layer.weight.data))
                elif prune_fn == functional.prune_batchnorm:
                    # regularize BN
                    if layer.affine is not None:
                        layer.weight.grad.data.add_(self.beta*torch.sign(layer.weight.data))
            

        