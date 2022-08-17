from .basepruner import MetaPruner, linear_scheduler
from .. import functional
import torch
import math

class GroupLassoPruner(MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        reg=0.1,
        pruning_steps=1,
        pruning_rate_scheduler=linear_scheduler,
        ch_sparsity=0.5,
        global_pruning=False,
        max_ch_sparsity=1.0,
        layer_ch_sparsity=None,
        round_to=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None
    ):
        super(GroupLassoPruner, self).__init__(
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
        self.plans = list(self.get_all_cliques())

    def regularize(self, model, loss):
        for clique in self.plans:
            group_norm = 0
            group_size = 0
            # Get group norm
            for dep, idxs in clique:
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn in [
                    functional.prune_conv_out_channel,
                    functional.prune_linear_out_channel,
                ]:
                    # regularize output channels
                    w = layer.weight.data[idxs].flatten(1)
                    group_size += w.shape[1]
                    group_norm += w.pow(2).sum(1)
                elif prune_fn in [
                    functional.prune_conv_in_channel,
                    functional.prune_linear_in_channel,
                ]:
                    # regularize input channels
                    w = layer.weight.data.transpose(0, 1)[idxs].flatten(1)
                    group_size += w.shape[1]
                    group_norm += w.pow(2).sum(1)
                elif prune_fn == functional.prune_batchnorm:
                    # regularize BN
                    if layer.affine is not None:
                        w = layer.weight.data[idxs]
                        group_size += w.shape[0]
                        group_norm += w.pow(2)

            group_norm = group_norm.sqrt()

            # Update Gradient
            for dep, idxs in clique:
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn in [
                    functional.prune_conv_out_channel,
                    functional.prune_linear_out_channel,
                ]:
                    # regularize output channels
                    w = layer.weight.data[idxs]
                    g = w / group_norm.view( -1, *([1]*(len(w.shape)-1)) )
                    layer.weight.grad[idxs].data.add_(self.reg*g)
                elif prune_fn in [
                    functional.prune_conv_in_channel,
                    functional.prune_linear_in_channel,
                ]:
                    # regularize input channels
                    w = layer.weight.data[:, idxs]
                    g = w / group_norm.view( 1, -1, *([1]*(len(w.shape)-2)) ) 
                    layer.weight.grad[:, idxs].data.add_(self.reg*g)
                elif prune_fn == functional.prune_batchnorm:
                    # regularize BN
                    if layer.affine is not None:
                        w = layer.weight.data[idxs]
                        g = w / group_norm
                        layer.weight.grad[idxs].data.add_(self.reg*g) 