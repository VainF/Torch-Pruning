from ..import function
from ... import ops, dependency
from .scheduler import linear_scheduler
import torch
import torch.nn as nn
import typing

class MetaPruner():
    """
        Meta Pruner for channel pruning.

        Args:
            model (class): the model to be pruned.
            example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
            importance (Callable): importance estimator.
            ch_sparsity (float): channel sparisty.
            ch_sparsity_dict (Dict[nn.Module, float]): user-specified layer sparsity.
            iterative_steps (int): number of pruning iterations.
            iterative_sparsity_scheduler (Callable): scheduler for iterative pruning.
            max_ch_sparsity (float): maximum channel sparsity.
            global_pruning (bool): enable global pruning.
            ignored_layers (List[nn.Module]): ignored modules.
            unwrapped_parameters (list): nn.Parameter defined by users
            output_transform (Callable): A function to transform network outputs.
            round_to (int): channel rounding.
        """

    def __init__(
        self,
        # Basic
        model: nn.Module,
        example_inputs: torch.Tensor,
        importance: typing.Callable,
        global_pruning: bool = False,
        ch_sparsity: float = 0.5, # channel/dim sparsity
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None,
        max_ch_sparsity: float = 1.0, 
        iterative_steps: int = 1, # for iterative pruning
        iterative_sparsity_scheduler: typing.Callable = linear_scheduler,
        ignored_layers: typing.List[nn.Module] = None, 

        # Advanced
        round_to: int = None, # round channels to 8x, 16x, ...
        channel_groups: typing.Dict[nn.Module, int] = dict(), # for grouped channels.
        customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None, # pruners for customized layers
        unwrapped_parameters: typing.List[nn.Parameter] = None, # unwrapped nn.Parameters like ViT.pos_emb
        root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM], # root module for each group
        output_transform: typing.Callable = None,
    ):
        self.model = model
        self.importance = importance
        self.ch_sparsity = ch_sparsity
        self.ch_sparsity_dict = ch_sparsity_dict if ch_sparsity_dict is not None else {}
        self.max_ch_sparsity = max_ch_sparsity
        self.global_pruning = global_pruning

        self.channel_groups = channel_groups
        self.root_module_types = root_module_types
        self.round_to = round_to

        # Build dependency graph
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            output_transform=output_transform,
            unwrapped_parameters=unwrapped_parameters,
            customized_pruners=customized_pruners,
        )

        self.ignored_layers = []  
        for layer in ignored_layers:
            self.ignored_layers.extend( list(layer.modules()) )

        self.iterative_steps = iterative_steps
        self.iterative_sparsity_scheduler = iterative_sparsity_scheduler
        self.current_step = 0

        # Record initial status
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        for m in self.DG.module2node.keys():
            if ops.module2type(m) in self.DG.REGISTERED_PRUNERS:
                self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
                self.layer_init_in_ch[m] = self.DG.get_in_channels(m)

        # global channel sparsity for each iterative step
        self.per_step_ch_sparsity = self.iterative_sparsity_scheduler(
            self.ch_sparsity, self.iterative_steps
        )
        
        # The customized channel sparsity for different layers
        self.ch_sparsity_dict = {}  
        if ch_sparsity_dict is not None:
            for module in ch_sparsity_dict:
                sparsity = ch_sparsity_dict[module]
                for submodule in module.modules():
                    prunable_types = [ ops.type2class(prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys() ]
                    if isinstance(submodule, prunable_types):
                        self.ch_sparsity_dict[submodule] = self.iterative_sparsity_scheduler(
                            sparsity, self.iterative_steps
                        )

        # detect group convs
        for m in self.model.modules():
            if isinstance(m, ops.TORCH_CONV) \
                and m.groups>1 \
                    and m.groups!=m.out_channels:
                        self.channel_groups[m] = m.groups
        
        if self.global_pruning:
            initial_total_channels = 0
            for group in self.get_all_groups():
                ch_groups = self.get_channel_groups(group)
                initial_total_channels += (self.DG.get_out_channels(group[0][0].target.module) // ch_groups) # utils.count_prunable_out_channels( group[0][0].target.module )
            self.initial_total_channels = initial_total_channels

    def get_target_sparsity(self, module):
        s = self.ch_sparsity_dict.get(module, self.per_step_ch_sparsity)[self.current_step]
        return min(s, self.max_ch_sparsity)

    def reset(self):
        self.current_step = 0

    def regularize(self, model, loss):
        """ Model regularizor
        """
        pass
    
    def get_all_groups(self):
        visited_layers = []
        for m in self.DG.module2node.keys():
            if m in self.ignored_layers:
                continue
            
            if not isinstance(m, tuple(self.root_module_types)):
                continue
        
            pruner = self.DG.REGISTERED_PRUNERS.get(ops.module2type(m), None)
            if pruner is None or pruner.get_out_channels(m) is None:
                continue

            if m in visited_layers:
                continue

            layer_channels = pruner.get_out_channels(m) 
            group = self.DG.get_pruning_group(m, pruner.prune_out_channels, list(range(layer_channels)))
            prunable_group = True
            for dep, _ in group:
                module = dep.target.module
                pruning_fn = dep.handler
                if function.is_out_channel_pruner(pruning_fn):
                    visited_layers.append(module)
                    if module in self.ignored_layers:
                        prunable_group = False
            if prunable_group:
                yield group

    def step(self):
        if self.global_pruning:
            self.prune_global()
        else:
            self.prune_local()
        self.current_step += 1

    def estimate_importance(self, group, ch_groups=1):
        return self.importance(group, ch_groups=ch_groups)

    def _check_sparsity(self, group):
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if function.is_out_channel_pruner(pruning_fn):
                target_sparsity = self.get_target_sparsity(module)
                layer_out_ch = self.DG.get_out_channels(module)
                
                if layer_out_ch < self.layer_init_out_ch[module] * (
                    1 - self.max_ch_sparsity
                ) or layer_out_ch==1:
                    return False

            elif function.is_in_channel_pruner(pruning_fn):
                #target_sparsity = self.get_target_sparsity(module)
                layer_in_ch = self.DG.get_in_channels(module)
                if layer_in_ch < self.layer_init_in_ch[module] * (
                    1 - self.max_ch_sparsity
                ) or layer_in_ch==1:
                    return False
        return True

    def get_channel_groups(self, group):
        if isinstance(self.channel_groups, int):
            return self.channel_groups
        for dep, _ in group:
            module = dep.target.module
            if module in self.channel_groups: #and function.is_out_channel_pruner(dep.handler):
                return self.channel_groups[module]
        return 1 # no channel grouping

    def prune_local(self):
        if self.current_step >= self.iterative_steps:
            return
        
        for group in self.get_all_groups():
            # check pruning rate
            if self._check_sparsity(group):
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler

                ch_groups = self.get_channel_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups)
                current_channels = self.DG.get_out_channels(module)
                target_sparsity = self.get_target_sparsity(module)
                n_pruned = current_channels - int(
                    self.layer_init_out_ch[module] *
                    (1 - target_sparsity)
                )

                if self.round_to:
                    n_pruned = n_pruned - (n_pruned % self.round_to)

                if n_pruned<=0: 
                    continue
                if ch_groups>1:
                    imp = imp[:len(imp)//ch_groups]
                imp_argsort = torch.argsort(imp)
                pruning_idxs = imp_argsort[:(n_pruned//ch_groups)]
                if ch_groups>1:
                    group_size = current_channels//ch_groups
                    pruning_idxs = torch.cat([pruning_idxs+group_size*i for i in range(ch_groups)], 0)
                group = self.DG.get_pruning_group(
                    module, pruning_fn, pruning_idxs.tolist())
                if self.DG.check_pruning_group(group):
                    group.exec()
            
    
    def prune_global(self):
        if self.current_step >= self.iterative_steps:
            return

        global_importance = []
        for group in self.get_all_groups():
            if self._check_sparsity(group):
                ch_groups = self.get_channel_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups)
                if ch_groups>1:
                    imp = imp[:len(imp)//ch_groups]
                global_importance.append((group, ch_groups, imp))

        imp = torch.cat([local_imp[-1] for local_imp in global_importance], dim=0)
        target_sparsity = self.per_step_ch_sparsity[self.current_step]
        n_pruned = len(imp) - int(
            self.initial_total_channels *
            (1 - target_sparsity)
        )
        if n_pruned<=0:
            return
        topk_imp, _ = torch.topk(imp, k=n_pruned, largest=False)

        # global pruning through thresholding
        thres = topk_imp[-1]
        for group, ch_groups, imp in global_importance:
            module = group[0][0].target.module
            pruning_fn = group[0][0].handler
            pruning_indices = (imp <= thres).nonzero().view(-1)
            if ch_groups>1:
                group_size = self.DG.get_out_channels(module)//ch_groups
                pruning_indices = torch.cat([pruning_indices+group_size*i for i in range(ch_groups)], 0)
            if self.round_to:
                n_pruned = len(pruning_indices)
                n_pruned = n_pruned - (n_pruned % self.round_to)
                pruning_indices = pruning_indices[:n_pruned]
            group = self.DG.get_pruning_group(module, pruning_fn, pruning_indices.tolist())
            if self.DG.check_pruning_group(group):
                group.exec()
    
