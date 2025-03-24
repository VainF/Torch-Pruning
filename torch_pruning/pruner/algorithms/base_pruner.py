import torch
import torch.nn as nn
import typing, warnings

from torch_pruning.pruner.importance import OBDCImportance

from .scheduler import linear_scheduler
from ..import function
from ... import ops, dependency
import math

class BasePruner:
    """
    Meta pruner for structural pruning.   
    It implements the group-level pruning strategy powered by Dependency Graph.  
    See https://arxiv.org/abs/2301.12900 for details.

    Args:
        model (nn.Module): A to-be-pruned model
        example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
        importance (Callable): importance estimator. 
        global_pruning (bool): enable global pruning. Default: False.
        pruning_ratio (float): global channel sparisty. Also known as pruning ratio. Default: 0.5.
        pruning_ratio_dict (Dict[nn.Module|Tuple[nn.Module], float]): layer-specific pruning ratio. Default: None. The key of the dict can be a single module or a tuple of modules. The pruning ratio will be shared by all modules in the tuple.
        max_pruning_ratio (float): the maximum pruning ratio. Default: 1.0.
        iterative_steps (int): number of steps for iterative pruning. Default: 1.
        iterative_pruning_ratio_scheduler (Callable): scheduler for iterative pruning. Default: linear_scheduler.
        ignored_layers (List[nn.Module | typing.Type]): ignored modules. Default: None.
        round_to (int): round channels to the nearest multiple of round_to. E.g., round_to=8 means channels will be rounded to 8x. Default: None.
        isomorphic (bool): enable isomorphic pruning. Default: False. https://arxiv.org/abs/2407.04616
        in_channel_groups (Dict[nn.Module, int]): The number of channel groups for layer input. Default: dict().
        out_channel_groups (Dict[nn.Module, int]): The number of channel groups for layer output. Default: dict().
        num_heads (Dict[nn.Module, int]): The number of heads for multi-head attention. Default: dict().
        prune_num_heads (bool): remove entire heads in multi-head attention. Default: False.
        prune_head_dims (bool): remove head dimensions in multi-head attention. Default: True.
        head_pruning_ratio (float): head pruning ratio. Default: 0.0.
        head_pruning_ratio_dict (Dict[nn.Module, float]): layer-specific head pruning ratio. Default: None.
        customized_pruners (dict): a dict containing module-pruner pairs. Default: None.
        unwrapped_parameters (dict): a dict containing unwrapped parameters & pruning dims. Default: None.
        root_module_types (list): types of prunable modules. Default: [nn.Conv2d, nn.Linear, nn.LSTM].
        forward_fn (Callable): A function to execute model.forward. Default: None.
        output_transform (Callable): A function to transform network outputs. Default: None.
        channel_groups (Dict[nn.Module, int]): output channel grouping. Default: dict().
        ch_sparsity (float): the same as pruning_ratio. Default: None.
        ch_sparsity_dict (Dict[nn.Module, float]): the same as pruning_ratio_dict. Default: None.
        """

    def __init__(
        self,
        # Basic
        model: nn.Module, # a simple pytorch model
        example_inputs: torch.Tensor, # a dummy input for graph tracing. Should be on the same 
        importance: typing.Callable, # tp.importance.Importance for group importance estimation
        global_pruning: bool = False, # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        pruning_ratio: float = 0.5,  # channel/dim pruning ratio, also known as pruning ratio
        pruning_ratio_dict: typing.Dict[typing.Union[nn.Module, typing.Tuple[nn.Module]], float] = None, # layer-specific pruning ratio. Will cover pruning_ratio if specified. The key of the dict can be a single module or a tuple of modules. The pruning ratio will be shared by all modules in the tuple. 
        max_pruning_ratio: float = 1.0, # maximum pruning ratio. useful if over-pruning happens.
        iterative_steps: int = 1,  # for iterative pruning
        iterative_pruning_ratio_scheduler: typing.Callable = linear_scheduler, # scheduler for iterative pruning.
        ignored_layers: typing.List[nn.Module] = None, # ignored layers
        round_to: int = None,  # round channels to the nearest multiple of round_to
        isomorphic: bool = False, # enable isomorphic pruning (ECCV 2024, https://arxiv.org/abs/2407.04616) if global_pruning=True. 

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
        channel_groups: typing.Dict[nn.Module, int] = dict(), # channel grouping
        ch_sparsity: float = None,
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None, 
    ):
        self.model = model
        self.importance = importance

        if ch_sparsity is not None:
            warnings.warn(
                "ch_sparsity is deprecated in v1.3.0. Please use pruning_ratio.")
            pruning_ratio = ch_sparsity
        if ch_sparsity_dict is not None:
            warnings.warn(
                "ch_sparsity_dict is deprecated in v1.3.0. Please use pruning_ratio_dict instead.")
            pruning_ratio_dict = ch_sparsity_dict

        self.pruning_ratio = pruning_ratio
        self.pruning_ratio_dict = pruning_ratio_dict if pruning_ratio_dict is not None else {}
        self.max_pruning_ratio = max_pruning_ratio
        self.global_pruning = global_pruning
        self.isomorphic = isomorphic

        if len(channel_groups) > 0:
            warnings.warn(
                "channel_groups is deprecated. Please use in_channel_groups and out_channel_groups instead.")
            out_channel_groups.update(channel_groups)

        if len(num_heads) > 0:
            out_channel_groups.update(num_heads)

        self.in_channel_groups = in_channel_groups
        self.out_channel_groups = out_channel_groups
        self.root_module_types = root_module_types
        self.round_to = round_to

        # MHA
        self.num_heads = num_heads
        self.prune_num_heads = prune_num_heads
        self.prune_head_dims = prune_head_dims
        self.head_pruning_ratio = head_pruning_ratio

        ###############################################
        # Ignored layers and submodules
        self.ignored_layers = []
        self.ignored_params = []
        if ignored_layers is not None:
            for layer in ignored_layers:
                if isinstance(layer, nn.Module):
                    self.ignored_layers.extend(list(layer.modules()))
                elif isinstance(layer, nn.Parameter):
                    self.ignored_params.append(layer)

        ###############################################
        # Build dependency graph
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            forward_fn=forward_fn,
            output_transform=output_transform,
            unwrapped_parameters=unwrapped_parameters,
            customized_pruners=customized_pruners,
            ignored_params=self.ignored_params,
        )

        ###############################################
        # Iterative pruning
        # The pruner will prune the model iteratively for several steps to achieve the target pruning ratio
        # E.g., if iterative_steps=5, pruning_ratio=0.5, the pruning ratio of each step will be [0.1, 0.2, 0.3, 0.4, 0.5]
        self.iterative_steps = iterative_steps
        self.iterative_pruning_ratio_scheduler = iterative_pruning_ratio_scheduler
        self.current_step = 0
        # channel pruning ratio for each iterative step
        self.per_step_pruning_ratio = self.iterative_pruning_ratio_scheduler(
            self.pruning_ratio, self.iterative_steps
        )
        self.per_step_head_pruning_ratio = self.iterative_pruning_ratio_scheduler(
            self.head_pruning_ratio, self.iterative_steps
        )

        ###############################################
        # Ranking Scopes
        # We will perform ranking within each scope.
        # If a scope only contains one layer, then we do local pruning
        # If a scope contains multiple layers, then global ranking will be applied to the entire scope
        # To manually specify the ranking scope, you can use pass a key-value pair to the pruning_ratio_dict, with a tuple of modules as the key.
        self._layer_to_scope = {}
        # initial channels for different scope. It will be filled during the first pruning step.
        self._scope_initial_channels = {}

        ###############################################
        # Layer-specific pruning ratios. Will cover the global ratio if specified
        # The key of the dict can be a single module or a tuple of modules. The pruning ratio will be shared by all modules in the tuple.
        self.pruning_ratio_dict = {}
        user_defined_scope_id = 0
        if pruning_ratio_dict is not None:
            for modules in pruning_ratio_dict:
                ratio = pruning_ratio_dict[modules]

                if isinstance(modules, tuple):
                    scope = modules  # will scan all modules sequentially
                else:
                    # only one model, do local pruning for this module
                    scope = [modules]

                scope_name = f"_User_Defined_Scope_{user_defined_scope_id}"
                local_pruning_scope_postfix = 0
                for m in scope:
                    for submodule in m.modules():
                        prunable_types = tuple([ops.type2class(
                            prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys()])
                        if isinstance(submodule, prunable_types):
                            if isinstance(submodule, nn.Module):
                                if not self.global_pruning:
                                    self._layer_to_scope[submodule] = (
                                        scope_name+f"_{local_pruning_scope_postfix}", scope)
                                    # assign each layer to a unique scope if local pruning
                                    local_pruning_scope_postfix += 1
                                else:
                                    # assign all layers to this scope
                                    self._layer_to_scope[submodule] = (
                                        scope_name, scope)

                            self.pruning_ratio_dict[submodule] = self.iterative_pruning_ratio_scheduler(
                                ratio, self.iterative_steps
                            )
                user_defined_scope_id += 1

        # Head pruning ratio
        self.head_pruning_ratio_dict = {}
        if head_pruning_ratio_dict is not None:
            for module in head_pruning_ratio_dict:
                ratio = head_pruning_ratio_dict[module]
                for submodule in module.modules():
                    prunable_types = tuple([ops.type2class(
                        prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys()])
                    if isinstance(submodule, prunable_types):
                        self.head_pruning_ratio_dict[submodule] = self.iterative_pruning_ratio_scheduler(
                            ratio, self.iterative_steps
                        )

        ###############################################
        # Detect group convs & group norms
        for m in self.model.modules():
            layer_pruner = self.DG.get_pruner_of_module(m)
            in_ch_group = layer_pruner.get_in_channel_groups(m)
            out_ch_group = layer_pruner.get_out_channel_groups(m)
            if isinstance(m, ops.TORCH_CONV) and m.groups == m.out_channels:
                continue
            if in_ch_group > 1:
                self.in_channel_groups[m] = in_ch_group
            if out_ch_group > 1:
                self.out_channel_groups[m] = out_ch_group

        ###############################################
        # Initial channels/dims of each layer
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        self.init_num_heads = {}
        for m in self.DG.module2node.keys():
            if ops.module2type(m) in self.DG.REGISTERED_PRUNERS:
                self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
                self.layer_init_in_ch[m] = self.DG.get_in_channels(m)
                if m in self.num_heads:
                    self.init_num_heads[m] = self.num_heads[m]

        ###############################################
        # Count the number of total channels at initialization
        # if self.global_pruning:
        initial_total_channels = 0
        initial_total_heads = 0
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types):
            _is_atten, qkv_layers = self._is_atten_group(group)
            if _is_atten:
                group = self._downstream_node_as_root_if_attention(group)
                if group is None:
                    continue
            initial_total_channels += ((self.DG.get_out_channels(
                group[0][0].target.module)) // self._get_channel_groups(group))
            for dep, _ in group:
                if dep.target.module in self.num_heads and self.DG.is_out_channel_pruning_fn(dep.handler):
                    initial_total_heads += self.num_heads[dep.target.module]
                    break  # only count heads once
        self.initial_total_channels = initial_total_channels
        self.initial_total_heads = initial_total_heads

    def step(self, interactive=False) -> typing.Union[typing.Generator, None]:
        self.current_step += 1
        if interactive:  # yield groups for interactive pruning
            return self._prune()
        else:
            for group in self._prune():
                group.prune()

    def manual_prune_width(self, layer, pruning_fn, pruning_ratios_or_idxs):
        if isinstance(pruning_ratios_or_idxs, float):
            if self.DG.is_out_channel_pruning_fn(pruning_fn):
                prunable_channels = self.DG.get_out_channels(layer)
            else:
                prunable_channels = self.DG.get_in_channels(layer)
            full_group = self.DG.get_pruning_group(
                layer, pruning_fn, list(range(prunable_channels)))
            imp = self.estimate_importance(full_group)
            imp_argsort = torch.argsort(imp)
            n_pruned = int(prunable_channels * (1 - pruning_ratios_or_idxs))
            pruning_idxs = imp_argsort[:n_pruned]

        group = self.DG.get_pruning_group(layer, pruning_fn, pruning_idxs)
        group.prune()

    def estimate_importance(self, group) -> torch.Tensor:
        return self.importance(group)

    def pruning_history(self) -> typing.List[typing.Tuple[str, bool, typing.Union[list, tuple]]]:
        return self.DG.pruning_history()

    def load_pruning_history(self, pruning_history) -> None:
        self.DG.load_pruning_history(pruning_history)

    def get_target_pruning_ratio(self, module, step=-1) -> float:
        if step < 0:
            step = self.current_step
        s = self.pruning_ratio_dict.get(
            module, self.per_step_pruning_ratio)[step]
        return min(s, self.max_pruning_ratio)

    def get_target_head_pruning_ratio(self, module) -> float:
        s = self.head_pruning_ratio_dict.get(module, self.per_step_head_pruning_ratio)[
            self.current_step]
        return min(s, 1)

    def reset(self) -> None:
        self.current_step = 0

    def update_regularizer(self) -> None:
        pass

    def regularize(self, model, loss) -> typing.Any:
        """ Model regularizer for sparse training
        """
        pass

    def _check_pruning_ratio(self, group) -> bool:
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if dep.target.type == ops.OPTYPE.PARAMETER:
                continue
            if self.DG.is_out_channel_pruning_fn(pruning_fn):
                layer_out_ch = self.DG.get_out_channels(module)
                if layer_out_ch is None:
                    continue
                if layer_out_ch < self.layer_init_out_ch[module] * (
                    1 - self.max_pruning_ratio
                ) or layer_out_ch == 1:
                    return False

            elif self.DG.is_in_channel_pruning_fn(pruning_fn):
                layer_in_ch = self.DG.get_in_channels(module)
                if layer_in_ch is None:
                    continue
                if layer_in_ch < self.layer_init_in_ch[module] * (
                    1 - self.max_pruning_ratio
                ) or layer_in_ch == 1:
                    return False
        return True

    def _is_atten_group(self, group) -> bool:
        is_attn = False
        qkv_layers = []
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if self.DG.is_out_channel_pruning_fn(pruning_fn) and module in self.num_heads:
                qkv_layers.append(module)
                is_attn = True
        return is_attn, qkv_layers

    def _get_channel_groups(self, group) -> int:
        ch_groups = []
        # has_unbind = False
        # unbind_node = None

        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            channel_groups = self.out_channel_groups if self.DG.is_out_channel_pruning_fn(
                pruning_fn) else self.in_channel_groups

            if module in channel_groups:
                ch_groups.append(channel_groups[module])

            # if dep.source.type==ops.OPTYPE.UNBIND:
            #    has_unbind = True
            #    unbind_node = dep.source

        # if has_unbind and ch_groups>1:
        #    ch_groups = ch_groups // len(unbind_node.outputs)
        if len(ch_groups) == 0:
            return 1
        return max(ch_groups)  # no channel grouping

    def _downstream_node_as_root_if_attention(self, group):
        # Use a downstream node as the root if torch.unbind exists. TODO: find a general way to handle torch.unbind in timm
        is_attention = False
        downstream_dep = None
        for _dep, _idxs in group:
            if _dep.source.module in self.num_heads and self.DG.is_out_channel_pruning_fn(_dep.handler):
                is_attention = True
            if isinstance(_dep.target.module, tuple(self.root_module_types)) and self.DG.is_in_channel_pruning_fn(_dep.handler):
                downstream_dep = _dep
                idxs = _idxs
        # use a downstream node as the root node for attention layers
        if is_attention and downstream_dep is not None:
            group = self.DG.get_pruning_group(
                downstream_dep.target.module, downstream_dep.handler, idxs)
            return group
        return None

    def _round_to(self, n_pruned, current_channels, round_to):
        rounded_channels = current_channels - n_pruned
        rounded_channels = rounded_channels - rounded_channels % round_to
        n_pruned = current_channels - rounded_channels
        return max(n_pruned, 0)

    @torch.no_grad()
    def _prune(self) -> typing.Generator:

        if self.current_step > self.iterative_steps:
            warnings.warn(
                "Pruning exceed the maximum iterative steps, no pruning will be performed.")
            return

        ##############################################
        # Initialize ranking scopes
        # A scope is a set of layers that will be considered as a basic unit during ranking. 
        # For example, for local pruning, each layer will be a scope.
        # This feature is useful for implementing ranking strategies such as local pruning, global pruning, 
        #   customized pruning ratios or isomorphic pruning (ECCV 2024): https://arxiv.org/abs/2407.04616
        # There are two pre-defined scopes: DEFAULT_SCOPE and ATTN_HEAD_SCOPE
        #   - DEFAULT_SCOPE: a group will be assigned to this scope for global ranking if not specified
        #   - ATTN_HEAD_SCOPE: for multi-head attention pruning
        ##############################################
        # Pre-defined scopes
        DEFAULT_SCOPE = "DEFAULT_SCOPE" # default scope for global pruning
        ATTN_HEAD_SCOPE = "ATTN_HEAD_SCOPE" # scope for multi-head attention pruning

        # ATTN_HEAD_SCOPE will be a dict, because we need to index these groups later.
        # Other scopes will be a simple list
        ranking_scope = {DEFAULT_SCOPE: [], ATTN_HEAD_SCOPE: {}}

        ##############################################
        # 1. Pre-compute importance for each group and assign them to different scopes
        ##############################################
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types):
            if self._check_pruning_ratio(group):
                # Re-order the nodes in a group and use a downstream node as the root for attention layers.
                # This will not change the group structure, but make index mapping easier for attention layers.
                _is_atten, qkv_layers = self._is_atten_group(group)
                if _is_atten:
                    group = self._downstream_node_as_root_if_attention(group)
                    if group is None:
                        continue
                
                ch_groups = self._get_channel_groups(group)
                imp = self.estimate_importance(group)  # raw importance score
                if imp is None:
                    continue
                    
                group_size = len(imp) // ch_groups
                # layers with dimension grouping, such as GroupConv, GroupNorm, Multi-head attention, etc.
                if ch_groups > 1:
                    # We average importance across groups here. For example:
                    # w = [1, 2, 3, 4, 5, 6] with groups=2.
                    # We have two groups [1,2,3] and [4,5,6].
                    # Those groups should have the same size after pruning,
                    # With the magnitude importance, we average the importance as [(|1|+|4|)/2, (|2|+|5|)/2, (|3|+|6|)/2] = [2.5, 3.5, 4.5]
                    # The remaining weights will be [2, 3, 5, 6]
                    dim_imp = imp.view(ch_groups, -1).mean(dim=0).cpu()
                else:
                    # no grouping
                    dim_imp = imp.cpu()

                # Importance scores for Attention Heads
                if _is_atten and self.prune_num_heads and self.get_target_head_pruning_ratio(qkv_layers[0]) > 0:
                    # average importance over heads
                    # Example: if we have the importance score:
                    # w = [1, 2, 3, 4, 5, 6] with num_heads=2
                    # Note: head1 = [1, 2, 3], head2 = [4, 5, 6]
                    # This is different from grouping. We need to remove the entire head.
                    # the average importance is [(1+2+3)/3, (4+5+6)/3] = [2, 5]
                    # So, the remaining heads will be [4, 5, 6]

                    # GQA: the number of heads for KV might be different from Q (Num_KV<=Num_Q)
                    # get the maximum number of heads
                    num_heads = max([self.num_heads[qkv_layer]
                                    for qkv_layer in qkv_layers])
                    # average importance by head.
                    head_imp = imp.view(num_heads, -1).mean(1).cpu()
                    ranking_scope[ATTN_HEAD_SCOPE][group] = (
                        qkv_layers, head_imp)

                # Scope Type 1: User-defined scope, such as layer-specific pruning_ratios
                # You can manually specify a pruning ratio to a specific layer or structure.
                # This will create an independent scope during pruning.
                is_user_defined_scope = False
                for dep, _ in group:
                    for module, pruning_fn in zip([dep.source.module, dep.target.module], [dep.trigger, dep.handler]):
                        if module in self._layer_to_scope and self.DG.is_out_channel_pruning_fn(pruning_fn):
                            scope_name, scope = self._layer_to_scope[module]
                            if len(scope) > 0:
                                pruning_ratio = self.get_target_pruning_ratio(
                                    module, step=self.current_step)
                                record = (group, ch_groups, group_size,
                                          pruning_ratio, dim_imp)
                                if scope_name not in ranking_scope:
                                    ranking_scope[scope_name] = []
                                ranking_scope[scope_name].append(record)
                                is_user_defined_scope = True
                        # A bit messy here. Will refactor in the future.
                        if is_user_defined_scope:
                            break
                    if is_user_defined_scope:
                        break
                if is_user_defined_scope:
                    continue

                # otherwise, use the default pruning ratio
                record = (group, ch_groups, group_size,
                          self.per_step_pruning_ratio[self.current_step], dim_imp)

                # Scope Type 2: Isomorphic Pruning
                if self.isomorphic:
                    scope_name = "Isomorphic_"
                    for dep, _ in group: 
                        # Check if two group have the same pruning patterns. 
                        # We transform the graph structure and pruning functions into a string tag for fast comparison
                        source = "%s_%s" % (type(
                            dep.source.module), "out" if self.DG.is_out_channel_pruning_fn(dep.handler) else "in")
                        target = "%s_%s" % (type(
                            dep.target.module), "out" if self.DG.is_out_channel_pruning_fn(dep.handler) else "in")
                        scope_name += "%s_%s" % (source, target)
                    if scope_name not in ranking_scope:
                        # New isomorphic group
                        ranking_scope[scope_name] = []
                    ranking_scope[scope_name].append(record)

                # Scope Type 3: use the default scope for global pruning
                elif self.global_pruning:  
                    ranking_scope[DEFAULT_SCOPE].append(record)

                # Scope Type 4: always create a new scope if local pruning
                else:  
                    module_name = self.DG._module2name[group[0]
                                                       [0].source.module]
                    ranking_scope[module_name] = [record]

        if len(ranking_scope[DEFAULT_SCOPE]) == 0 and len(ranking_scope[ATTN_HEAD_SCOPE]) == 0 and len(ranking_scope) <= 2:
            return

        ##############################################
        # 2. Thresholding by ranking all importance scores within each scope
        ##############################################

        # 2.1 Compute the threshold for global attn head pruning
        if len(ranking_scope[ATTN_HEAD_SCOPE]) > 0 and self.global_pruning:
            concat_head_imp = torch.cat(
                [local_imp[-1] for local_imp in ranking_scope[ATTN_HEAD_SCOPE].values()], dim=0)
            target_head_pruning_ratio = self.per_step_head_pruning_ratio[self.current_step]
            n_heads_removed = len(concat_head_imp) - int(
                self.initial_total_heads *
                (1 - target_head_pruning_ratio)
            )
            if n_heads_removed > 0:
                topk_head_imp, _ = torch.topk(
                    concat_head_imp, k=n_heads_removed, largest=False)
                head_thres = topk_head_imp[-1]

        # 2.2 Width pruning, including channels, hidden dims, etc.
        width_pruning_scope_names = [
            k for k in ranking_scope.keys() if k != ATTN_HEAD_SCOPE]
        for scope_id, scope_name in enumerate(width_pruning_scope_names):
            #if not self.global_pruning:
            #    assert len(
            #        ranking_scope[scope_name]) <= 1, "Internal Error: local pruning should only contain less than one layer per scope."

            # records[i] -> (group, ch_groups, group_size, pruning_ratio, dim_imp)_i
            records = ranking_scope[scope_name]
            # Find the threshold for pruning
            if len(records) > 0:
                # concatenate importance scores in this scope
                concat_imp = torch.cat([local_imp[-1]
                                       for local_imp in records], dim=0)
                # records[i] -> (group, ch_groups, group_size, pruning_ratio, dim_imp)_i
                target_pruning_ratio = records[0][-2]
                if scope_name not in self._scope_initial_channels:
                    self._scope_initial_channels[scope_name] = len(concat_imp)

                n_pruned = len(concat_imp) - int(
                    self._scope_initial_channels[scope_name] *
                    (1 - target_pruning_ratio)
                )

                if n_pruned > 0:
                    topk_imp, topk_indices = torch.topk(
                        concat_imp, k=n_pruned, largest=False)
                    thres = topk_imp[-1]

                    # Perform pruning in each scope
                    for group, ch_groups, group_size, target_pruning_ratio, imp in records:
                        module = group[0].dep.target.module
                        pruning_fn = group[0].dep.handler
                        get_channel_fn = self.DG.get_out_channels if self.DG.is_out_channel_pruning_fn(
                            pruning_fn) else self.DG.get_in_channels
                        _is_atten, qkv_layers = self._is_atten_group(group)

                        # Prune dims/channels
                        pruning_indices = []
                        if not _is_atten or self.prune_head_dims:
                            if self.global_pruning:
                                _pruning_indices = (
                                    imp <= thres).nonzero().view(-1)
                            else:
                                _pruning_indices = topk_indices
                            imp_argsort = torch.argsort(imp)
                            # recompute the number of pruned channels if round_to is enabled
                            if len(_pruning_indices) > 0 and self.round_to:
                                n_pruned = len(_pruning_indices)
                                current_channels = get_channel_fn(module)
                                n_pruned = self._round_to(
                                    n_pruned, current_channels, self.round_to)
                                _pruning_indices = imp_argsort[:n_pruned]
                            if ch_groups > 1:  
                                # if channel grouping is enabled, we repeat the pruning indices for each channel group.
                                # For example, w=[0,1,2,3,4,5,6,7,8] with groups=3, and the pruning indices are [0].
                                # We extend the indices as [0, 3, 6] to remove the first element in each group.
                                for g_id in range(ch_groups):
                                    pruning_indices.append(
                                        _pruning_indices+g_id*group_size)
                            else:
                                pruning_indices.append(_pruning_indices)

                        # Check if this is an Attention that requires head pruning 
                        if len(ranking_scope[ATTN_HEAD_SCOPE]) > 0:
                            if group in ranking_scope[ATTN_HEAD_SCOPE]:
                                qkv_layers, head_imp = ranking_scope[ATTN_HEAD_SCOPE][group]
                                num_heads = max([self.num_heads[qkv_layer]
                                                for qkv_layer in qkv_layers])
                                _is_gqa = not all(
                                    [self.num_heads[qkv_layer] == num_heads for qkv_layer in qkv_layers])

                                if not self.global_pruning:  # local pruning
                                    n_heads_removed_per_group = math.ceil(
                                        self.get_target_head_pruning_ratio(qkv_layers[0]) * len(head_imp))
                                    if not _is_gqa:
                                        head_pruning_indices = torch.topk(
                                            head_imp, k=n_heads_removed_per_group, largest=False)[1]  # local ranking
                                    else:  # chunk the head imp
                                        num_kv_heads = min(
                                            [self.num_heads[qkv_layer] for qkv_layer in qkv_layers])
                                        num_heads = max(
                                            [self.num_heads[qkv_layer] for qkv_layer in qkv_layers])
                                        n_heads_removed_per_group = math.ceil(
                                            n_heads_removed_per_group / num_kv_heads)
                                        head_pruning_indices = []
                                        for kv_head_id in range(num_kv_heads):
                                            head_imp_kv = head_imp[kv_head_id * num_heads//num_kv_heads: (
                                                kv_head_id+1) * num_heads//num_kv_heads]
                                            head_pruning_indices_kv = torch.topk(
                                                head_imp_kv, k=n_heads_removed_per_group, largest=False)[1]
                                            head_pruning_indices.append(
                                                head_pruning_indices_kv + kv_head_id*num_heads//num_kv_heads)
                                        head_pruning_indices = torch.cat(
                                            head_pruning_indices, 0)
                                else:  # global pruning
                                    head_pruning_indices = (
                                        head_imp <= head_thres).nonzero().view(-1)  # global ranking
                                    if _is_gqa:
                                        num_kv_heads = min(
                                            [self.num_heads[qkv_layer] for qkv_layer in qkv_layers])
                                        n_heads_removed_per_group = math.ceil(
                                            len(head_pruning_indices) / num_kv_heads)
                                        head_pruning_indices = []
                                        for kv_head_id in range(num_kv_heads):
                                            head_imp_kv = head_imp[kv_head_id * len(head_imp)//num_kv_heads: (
                                                kv_head_id+1) * len(head_imp)//num_kv_heads]
                                            head_pruning_indices_kv = torch.topk(
                                                head_imp_kv, k=n_heads_removed_per_group, largest=False)[1]
                                            head_pruning_indices.append(
                                                head_pruning_indices_kv + kv_head_id*num_kv_heads)
                                        head_pruning_indices = torch.cat(
                                            head_pruning_indices, 0)

                                if len(head_pruning_indices) > 0:
                                    if len(qkv_layers) == 1:
                                        head_dim = qkv_layers[0].out_features // (
                                            self.num_heads[qkv_layers[0]]*3)
                                    else:
                                        head_dim = qkv_layers[0].out_features // self.num_heads[qkv_layers[0]]

                                    for head_id in head_pruning_indices:
                                        pruning_indices.append(torch.arange(
                                            head_id*head_dim, (head_id+1)*head_dim, device=head_imp.device))

                                num_heads = max([self.num_heads[qkv_layer]
                                                for qkv_layer in qkv_layers])
                                for qkv_layer in qkv_layers:
                                    if self.num_heads[qkv_layer] == num_heads:
                                        # update num heads after pruning
                                        self.num_heads[qkv_layer] -= len(
                                            head_pruning_indices)
                                        # update out_channel_groups
                                        self.out_channel_groups[qkv_layer] = self.num_heads[qkv_layer]

                        if len(pruning_indices) == 0:
                            continue
                        pruning_indices = torch.unique(
                            torch.cat(pruning_indices, 0)).tolist()

                        if isinstance(self.importance, OBDCImportance):
                            self.importance.adjust_fisher(
                                group, pruning_indices)

                        # create pruning group
                        group = self.DG.get_pruning_group(
                            module, pruning_fn, pruning_indices)
                    
                        if _is_atten:
                            _is_gqa = not all(
                                [self.num_heads[qkv_layer] == self.num_heads[qkv_layers[0]] for qkv_layer in qkv_layers])
                            if _is_gqa and self.prune_num_heads:
                                num_kv_heads = min(
                                    [self.num_heads[qkv_layer] for qkv_layer in qkv_layers])
                                kv_layers = [
                                    qkv_layer for qkv_layer in qkv_layers if self.num_heads[qkv_layer] == num_kv_heads]
                                for i in range(len(group)):
                                    dep, idxs = group[i]
                                    if dep.target.module in kv_layers:
                                        # disable head pruning for the kv layers if GQA is enabled, since they will be shared by multiple Q heads
                                        group[i] = (dep, [])

                        if self.DG.check_pruning_group(group):
                            yield group  # yield the group for interactive pruning
