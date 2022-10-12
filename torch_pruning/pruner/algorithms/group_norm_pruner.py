from .metapruner import MetaPruner
from .scheduler import linear_scheduler
from .. import function
import torch
import math
from ..._helpers import _FlattenIndexTransform

class GroupNormPruner(MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        reg=1e-4,
        iterative_steps=1,
        iterative_sparsity_scheduler=linear_scheduler,
        ch_sparsity=0.5,
        global_pruning=False,
        channel_groups=dict(),
        max_ch_sparsity=1.0,
        ch_sparsity_dict=None,
        round_to=None,
        ignored_layers=None,
        customized_pruners=None,
        unwrapped_parameters=None,
        output_transform=None,
    ):
        super(GroupNormPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            iterative_steps=iterative_steps,
            iterative_sparsity_scheduler=iterative_sparsity_scheduler,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict,
            global_pruning=global_pruning,
            channel_groups=channel_groups,
            max_ch_sparsity=max_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            output_transform=output_transform,
        )
        self.reg = reg
        self.groups = list(self.get_all_groups())

    @torch.no_grad()
    def regularize(self, model):
        gnorm_list = []

        for i, group in enumerate(self.groups):
            ch_groups = self.get_channel_groups(group)
            group_norm = 0
            # Get group norm
            for dep, idxs in group:
                idxs.sort()
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    # regularize output channels
                    w = layer.weight.data[idxs].flatten(1)
                    group_norm += w.pow(2).sum(1)

                elif prune_fn in [
                    function.prune_conv_in_channels,
                    function.prune_linear_in_channels,
                ]:
                    #w = (layer.weight)[:, idxs].transpose(0, 1).flatten(1)
                    w = (layer.weight).transpose(0, 1).flatten(1)
                    if (
                        w.shape[0] != group_norm.shape[0] and ch_groups==1
                    ):  # for conv-flatten 
    
                        if hasattr(dep.target, 'index_transform') and isinstance(dep.target.index_transform, _FlattenIndexTransform):
                            w = w.view(
                            group_norm.shape[0],
                            w.shape[0] // group_norm.shape[0],
                            w.shape[1],
                        ).flatten(1)
                        else:
                            w = w.view( w.shape[0] // group_norm.shape[0], group_norm.shape[0], w.shape[1] ).transpose(0, 1).flatten(1)

                    local_norm = w.abs().pow(2).sum(1)
                    if ch_groups>1:
                        local_norm = local_norm.repeat(ch_groups)
                    group_norm += local_norm[idxs]
                elif prune_fn == function.prune_batchnorm_out_channels:
                    # regularize BN
                    if layer.affine is not None:
                        w = layer.weight.data[idxs]
                        group_norm += w.pow(2)
            current_channels = len(group_norm)
            if ch_groups>1:
                group_norm = group_norm.view(ch_groups, -1).sum(0)
                group_stride = current_channels//ch_groups
                group_norm = torch.cat([group_norm+group_stride*i for i in range(ch_groups)], 0)
            group_norm = group_norm.sqrt()

            # Update Gradient
            for dep, idxs in group:
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    # regularize output channels
                    w = layer.weight.data[idxs]
                    g = w / group_norm.view( -1, *([1]*(len(w.shape)-1)) ) #* group_size #* scale.view( -1, *([1]*(len(w.shape)-1)) )
                    layer.weight.grad.data[idxs]+=self.reg * g 
                elif prune_fn in [
                    function.prune_conv_in_channels,
                    function.prune_linear_in_channels,
                ]:
                    w = layer.weight.data[:, idxs]
                    gn = group_norm
                    if (
                        w.shape[1] != group_norm.shape[0]
                    ):  # for conv-flatten 
                        if hasattr(dep.target, 'index_transform') and isinstance(dep.target.index_transform, _FlattenIndexTransform):
                            gn = group_norm.repeat_interleave(w.shape[1]//group_norm.shape[0])
                        else:
                            gn = group_norm.repeat(w.shape[1]//group_norm.shape[0])
                    # regularize input channels
                    w = layer.weight.data[:, idxs]
                    g = w / gn.view( 1, -1, *([1]*(len(w.shape)-2)) ) #* group_size  #* scale.view( 1, -1, *([1]*(len(w.shape)-2)) )
                    layer.weight.grad.data[:, idxs]+=self.reg * g
                elif prune_fn == function.prune_batchnorm_out_channels:
                    # regularize BN
                    if layer.affine is not None:
                        w = layer.weight.data[idxs]
                        g = w / group_norm
                        layer.weight.grad.data[idxs]+=self.reg * g 
        return gnorm_list