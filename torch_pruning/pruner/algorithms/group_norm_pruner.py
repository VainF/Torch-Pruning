import torch
import math
from .metapruner import MetaPruner
from .scheduler import linear_scheduler
from .. import function
from ..._helpers import _FlattenIndexMapping


class GroupNormPruner(MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        reg=1e-4,
        alpha=4,
        iterative_steps=1,
        iterative_sparsity_scheduler=linear_scheduler,
        ch_sparsity=0.5,
        global_pruning=False,
        channel_groups=dict(),
        max_ch_sparsity=1.0,
        soft_keeping_ratio=0.0,
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
        self.alpha = alpha
        self.groups = list(self.DG.get_all_groups())
        self.soft_keeping_ratio = soft_keeping_ratio
        self.cnt = 0

    @torch.no_grad()
    def regularize(self, model, alpha=16):
        for i, group in enumerate(self.groups):
            ch_groups = self.get_channel_groups(group)
            imp = self.estimate_importance(group).sqrt()
            gamma = alpha**((imp.max() - imp) / (imp.max() - imp.min()))

            # Update Gradient
            for dep, idxs in group:
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    w = layer.weight.data[idxs]
                    g = w * gamma.view( -1, *([1]*(len(w.shape)-1)) ) #/ group_norm.view( -1, *([1]*(len(w.shape)-1)) ) * group_size #group_size #* gamma.view( -1, *([1]*(len(w.shape)-1)) )
                    layer.weight.grad.data[idxs]+=self.reg * g 
                    
                    if layer.bias is not None:
                        b = layer.bias.data[idxs]
                        g = b * gamma
                        layer.bias.grad.data[idxs]+=self.reg * g 

                elif prune_fn in [
                    function.prune_conv_in_channels,
                    function.prune_linear_in_channels,
                ]:
                    gn = imp
                    if hasattr(dep.target, 'index_transform') and isinstance(dep.target.index_transform, _FlattenIndexMapping):
                        gn = imp.repeat_interleave(w.shape[1]//imp.shape[0])
                    
                    # regularize input channels
                    if prune_fn==function.prune_conv_in_channels and layer.groups>1:
                        gamma = gamma[:len(idxs)//ch_groups]
                        idxs = idxs[:len(idxs)//ch_groups]

                    w = layer.weight.data[:, idxs]
                    g = w * gamma.view( 1, -1, *([1]*(len(w.shape)-2))  ) #/ gn.view( 1, -1, *([1]*(len(w.shape)-2)) ) * group_size #* gamma.view( 1, -1, *([1]*(len(w.shape)-2))  )
                    layer.weight.grad.data[:, idxs]+=self.reg * g
                    
                elif prune_fn == function.prune_batchnorm_out_channels:
                    # regularize BN
                    if layer.affine is not None:
                        w = layer.weight.data[idxs]
                        g = w * gamma #/ group_norm * group_size
                        layer.weight.grad.data[idxs]+=self.reg * g 
                        
                        b = layer.bias.data[idxs]
                        g = b * gamma #/ group_norm * group_size
                        layer.bias.grad.data[idxs]+=self.reg * g 
        self.cnt+=1