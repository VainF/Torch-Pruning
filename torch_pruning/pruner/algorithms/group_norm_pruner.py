from .metapruner import MetaPruner
from .scheduler import linear_scheduler
from .. import function
import torch
import math
from ..._helpers import _FlattenIndexMapping

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
        self.groups = list(self.get_all_groups())
        self.soft_keeping_ratio = soft_keeping_ratio
        self.cnt = 0

    @torch.no_grad()
    def regularize(self, model, base=16):
        for i, group in enumerate(self.groups):
            ch_groups = self.get_channel_groups(group)
            group_norm = 0

            # Get group norm
            #print(group)
            for dep, idxs in group:
                idxs.sort()
                layer = dep.target.module
                prune_fn = dep.handler

                # Conv out_channels
                if prune_fn in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    w = layer.weight.data[idxs].flatten(1)
                    local_norm = w.pow(2).sum(1)
                    #print(local_norm.shape, layer, idxs, ch_groups)
                    if ch_groups>1:
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                        local_norm = local_norm.repeat(ch_groups)
                    group_norm+=local_norm
                    #if layer.bias is not None:
                    #    group_norm += layer.bias.data[idxs].pow(2)
                # Conv in_channels
                elif prune_fn in [
                    function.prune_conv_in_channels,
                    function.prune_linear_in_channels,
                ]:
                    w = (layer.weight).transpose(0, 1).flatten(1)
                    if (
                        w.shape[0] != group_norm.shape[0]
                    ):  
                        if hasattr(dep, 'index_mapping') and isinstance(dep.index_mapping, _FlattenIndexMapping):
                            # conv - latten
                            w = w.view(
                                group_norm.shape[0],
                                w.shape[0] // group_norm.shape[0],
                                w.shape[1],
                            ).flatten(1)
                        elif ch_groups>1 and prune_fn==function.prune_conv_in_channels and layer.groups==1:
                            # group conv
                            w = w.view(w.shape[0] // group_norm.shape[0],
                                    group_norm.shape[0], w.shape[1]).transpose(0, 1).flatten(1)               
                    local_norm = w.pow(2).sum(1)
                    if ch_groups>1:
                        if len(local_norm)==len(group_norm):
                            local_norm = local_norm.view(ch_groups, -1).sum(0)
                        local_norm = local_norm.repeat(ch_groups)
                    group_norm += local_norm[idxs]
                # BN
                elif prune_fn == function.prune_batchnorm_out_channels:
                    # regularize BN
                    if layer.affine:
                        w = layer.weight.data[idxs]
                        local_norm = w.pow(2)
                        if ch_groups>1:
                            local_norm = local_norm.view(ch_groups, -1).sum(0)
                            local_norm = local_norm.repeat(ch_groups)
                        group_norm += local_norm

                        #b = layer.bias.data[idxs]
                        #local_norm = b.pow(2)
                        #if ch_groups>1:
                        #    local_norm = local_norm.view(ch_groups, -1).sum(0)
                        #    local_norm = local_norm.repeat(ch_groups)
                        #group_norm += local_norm


            current_channels = len(group_norm)
            if ch_groups>1:
                group_norm = group_norm.view(ch_groups, -1).sum(0)
                group_stride = current_channels//ch_groups
                group_norm = torch.cat([group_norm+group_stride*i for i in range(ch_groups)], 0)
            group_norm = group_norm.sqrt()
            base = 16
            scale = base**((group_norm.max() - group_norm) / (group_norm.max() - group_norm.min()))
            #if self.cnt%1000==0:
            #    print("="*15)
            #    print(group)
            #    print("Group {}".format(i))
            #    print(group_norm)
            #    print(scale)
            
            # Update Gradient
            for dep, idxs in group:
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    w = layer.weight.data[idxs]
                    g = w * scale.view( -1, *([1]*(len(w.shape)-1)) ) #/ group_norm.view( -1, *([1]*(len(w.shape)-1)) ) * group_size #group_size #* scale.view( -1, *([1]*(len(w.shape)-1)) )
                    layer.weight.grad.data[idxs]+=self.reg * g 
                    #if layer.bias is not None:
                    #    b = layer.bias.data[idxs]
                    #    g = b * scale
                    #    layer.bias.grad.data[idxs]+=self.reg * g 
                elif prune_fn in [
                    function.prune_conv_in_channels,
                    function.prune_linear_in_channels,
                ]:
                    gn = group_norm
                    if hasattr(dep.target, 'index_transform') and isinstance(dep.target.index_transform, _FlattenIndexTransform):
                        gn = group_norm.repeat_interleave(w.shape[1]//group_norm.shape[0])
                    # regularize input channels
                    if prune_fn==function.prune_conv_in_channels and layer.groups>1:
                        scale = scale[:len(idxs)//ch_groups]
                        idxs = idxs[:len(idxs)//ch_groups]
                    w = layer.weight.data[:, idxs]
                    g = w * scale.view( 1, -1, *([1]*(len(w.shape)-2))  ) #/ gn.view( 1, -1, *([1]*(len(w.shape)-2)) ) * group_size #* scale.view( 1, -1, *([1]*(len(w.shape)-2))  )
                    layer.weight.grad.data[:, idxs]+=self.reg * g
                elif prune_fn == function.prune_batchnorm_out_channels:
                    # regularize BN
                    if layer.affine is not None:
                        w = layer.weight.data[idxs]
                        g = w * scale #/ group_norm * group_size
                        layer.weight.grad.data[idxs]+=self.reg * g 

                        #b = layer.bias.data[idxs]
                        #g = b * scale #/ group_norm * group_size
                        #layer.bias.grad.data[idxs]+=self.reg * g 
        self.cnt+=1