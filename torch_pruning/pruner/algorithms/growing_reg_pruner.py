import torch
import torch.nn as nn
import typing 
from .metapruner import MetaPruner
from .scheduler import linear_scheduler
from .. import function
from ... import ops

class GrowingRegPruner(MetaPruner):
    """ pruning with growing regularization
    https://arxiv.org/abs/2012.09243

    Args:

            # Basic
            * model (nn.Module): A to-be-pruned model
            * example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
            * importance (Callable): importance estimator. 
            * reg (float): regularization coefficient. Default: 1e-5.
            * delta_reg (float): increment of regularization coefficient. Default: 1e-5.
            * global_pruning (bool): enable global pruning. Default: False.
            * pruning_ratio (float): global channel sparisty. Also known as pruning ratio. Default: 0.5.
            * pruning_ratio_dict (Dict[nn.Module, float]): layer-specific pruning ratio. Will cover pruning_ratio if specified. Default: None.
            * max_pruning_ratio (float): the maximum pruning ratio. Default: 1.0.
            * iterative_steps (int): number of steps for iterative pruning. Default: 1.
            * iterative_pruning_ratio_scheduler (Callable): scheduler for iterative pruning. Default: linear_scheduler.
            * ignored_layers (List[nn.Module | typing.Type]): ignored modules. Default: None.
            * round_to (int): round channels to the nearest multiple of round_to. E.g., round_to=8 means channels will be rounded to 8x. Default: None.
            
            # Adavanced
            * in_channel_groups (Dict[nn.Module, int]): The number of channel groups for layer input. Default: dict().
            * out_channel_groups (Dict[nn.Module, int]): The number of channel groups for layer output. Default: dict().
            * num_heads (Dict[nn.Module, int]): The number of heads for multi-head attention. Default: dict().
            * prune_num_heads (bool): remove entire heads in multi-head attention. Default: False.
            * prune_head_dims (bool): remove head dimensions in multi-head attention. Default: True.
            * head_pruning_ratio (float): head pruning ratio. Default: 0.0.
            * head_pruning_ratio_dict (Dict[nn.Module, float]): layer-specific head pruning ratio. Default: None.
            * customized_pruners (dict): a dict containing module-pruner pairs. Default: None.
            * unwrapped_parameters (dict): a dict containing unwrapped parameters & pruning dims. Default: None.
            * root_module_types (list): types of prunable modules. Default: [nn.Conv2d, nn.Linear, nn.LSTM].
            * forward_fn (Callable): A function to execute model.forward. Default: None.
            * output_transform (Callable): A function to transform network outputs. Default: None.

            # Deprecated
            * channel_groups (Dict[nn.Module, int]): output channel grouping. Default: dict().
            * ch_sparsity (float): the same as pruning_ratio. Default: None.
            * ch_sparsity_dict (Dict[nn.Module, float]): the same as pruning_ratio_dict. Default: None.
    """
    def __init__(
        self,
        # Basic
        model: nn.Module, # a simple pytorch model
        example_inputs: torch.Tensor, # a dummy input for graph tracing. Should be on the same 
        importance: typing.Callable, # tp.importance.Importance for group importance estimation
        reg=1e-5, # regularization coefficient
        delta_reg=1e-5, # increment of regularization coefficient
        global_pruning: bool = False, # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        pruning_ratio: float = 0.5,  # channel/dim pruning ratio, also known as pruning ratio
        pruning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific pruning ratio, will cover pruning_ratio if specified
        max_pruning_ratio: float = 1.0, # maximum pruning ratio. useful if over-pruning happens.
        iterative_steps: int = 1,  # for iterative pruning
        iterative_pruning_ratio_scheduler: typing.Callable = linear_scheduler, # scheduler for iterative pruning.
        ignored_layers: typing.List[nn.Module] = None, # ignored layers
        round_to: int = None,  # round channels to the nearest multiple of round_to

        # Advanced
        in_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer input
        out_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer output
        num_heads: typing.Dict[nn.Module, int] = dict(), # The number of heads for multi-head attention
        prune_num_heads: bool = False, # remove entire heads in multi-head attention
        prune_head_dims: bool = True, # remove head dimensions in multi-head attention
        head_pruning_ratio: float = 0.0, # head pruning ratio
        head_pruning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific head pruning ratio
        customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None, # pruners for customized layers. E.g., {nn.Linear: my_linear_pruner}
        unwrapped_parameters: typing.Dict[nn.Parameter, int] = None, # unwrapped nn.Parameters & pruning_dims. For example, {ViT.pos_emb: 0}
        root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        forward_fn: typing.Callable = None, # a function to execute model.forward
        output_transform: typing.Callable = None, # a function to transform network outputs

        # deprecated
        channel_groups: typing.Dict[nn.Module, int] = dict(), # channel groups for layers
        ch_sparsity: float = None,
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None, 
    ):
        super(GrowingRegPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            global_pruning=global_pruning,
            pruning_ratio=pruning_ratio,
            pruning_ratio_dict=pruning_ratio_dict,
            max_pruning_ratio=max_pruning_ratio,
            iterative_steps=iterative_steps,
            iterative_pruning_ratio_scheduler=iterative_pruning_ratio_scheduler,
            ignored_layers=ignored_layers,
            round_to=round_to,
            
            in_channel_groups=in_channel_groups,
            out_channel_groups=out_channel_groups,
            num_heads=num_heads,
            prune_num_heads=prune_num_heads,
            prune_head_dims=prune_head_dims,
            head_pruning_ratio=head_pruning_ratio,
            head_pruning_ratio_dict=head_pruning_ratio_dict,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            root_module_types=root_module_types,
            forward_fn=forward_fn,
            output_transform=output_transform,
            
            channel_groups=channel_groups,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict
        )
        self.base_reg = reg
        self._groups = list(self.DG.get_all_groups(root_module_types=self.root_module_types, ignored_layers=self.ignored_layers))

        group_reg = {}
        for group in self._groups:
            group_reg[group] = torch.ones(len(group[0].idxs)) * self.base_reg
        self.group_reg = group_reg
        self.delta_reg = delta_reg

    def update_reg(self):
        for group in self._groups:
            group_l2norm_sq = self.estimate_importance(group)
            if group_l2norm_sq is None:
                continue
            reg = self.group_reg[group]
            standarized_imp = (group_l2norm_sq.max() - group_l2norm_sq) / \
                (group_l2norm_sq.max() - group_l2norm_sq.min() + 1e-8)  # => [0, 1]
            reg = reg + self.delta_reg * standarized_imp.to(reg.device)
            self.group_reg[group] = reg

    def update_regularizor(self):
        # Update the group list after pruning
        self._groups = list(self.DG.get_all_groups(root_module_types=self.root_module_types, ignored_layers=self.ignored_layers))
        group_reg = {}
        for group in self._groups:
            group_reg[group] = torch.ones(len(group[0].idxs)) * self.base_reg
        self.group_reg = group_reg

    def regularize(self, model, bias=False):
        for i, group in enumerate(self._groups):
            group_l2norm_sq = self.estimate_importance(group)
            if group_l2norm_sq is None or torch.any(torch.isnan(group_l2norm_sq)): # avoid nan
                continue
            gamma = self.group_reg[group]
            for k, (dep, idxs) in enumerate(group):
                layer = dep.layer
                pruning_fn = dep.pruning_fn

                if isinstance(layer, nn.modules.batchnorm._BatchNorm) and layer.affine == True and layer not in self.ignored_layers:
                    if layer.weight.grad is None: continue

                    root_idxs = group[k].root_idxs
                    _gamma = torch.index_select(gamma, 0, torch.tensor(root_idxs, device=gamma.device))
                    
                    layer.weight.grad.data[idxs].add_(_gamma.to(layer.weight.device) * layer.weight.data[idxs])
                    if bias and layer.bias is not None:
                        layer.bias.grad.data[idxs].add_(_gamma.to(layer.weight.device) * layer.bias.data[idxs])
                elif isinstance(layer, (nn.modules.conv._ConvNd, nn.Linear)):
                    if pruning_fn in [function.prune_conv_out_channels, function.prune_linear_out_channels] and layer not in self.ignored_layers:
                        if layer.weight.grad is None: continue

                        root_idxs = group[k].root_idxs
                        _gamma = torch.index_select(gamma, 0, torch.tensor(root_idxs, device=gamma.device))

                        w = layer.weight.data[idxs]
                        g = w * _gamma.to(layer.weight.device).view(-1, *([1]*(len(w.shape)-1)))
                        layer.weight.grad.data[idxs] += g

                        if bias and layer.bias is not None:
                            b = layer.bias.data[idxs]
                            g = b * _gamma.to(layer.weight.device)
                            layer.bias.grad.data[idxs] += g
                    elif pruning_fn in [function.prune_conv_in_channels, function.prune_linear_in_channels]:
                        if layer.weight.grad is None: continue
                        root_idxs = group[k].root_idxs
                        _gamma = torch.index_select(gamma, 0, torch.tensor(root_idxs, device=gamma.device))

                        w = layer.weight.data[:, idxs]
                        g = w * _gamma.to(layer.weight.device).view(1, -1, *([1]*(len(w.shape)-2)))
                        layer.weight.grad.data[:, idxs] += g
