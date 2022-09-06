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
            layer_ch_sparsity (Dict[nn.Module, float]): user-specified layer sparsity.
            pruning_steps (int): number of pruning iterations.
            pruning_rate_scheduler (Callable): scheduler for iterative pruning.
            max_ch_sparsity (float): maximum channel sparsity.
            global_pruning (bool): enable global pruning.
            ignored_layers (List[nn.Module]): ignored modules.
            user_defined_parameters (list): nn.Parameter defined by users
            output_transform (Callable): A function to transform network outputs.
            round_to (int): channel rounding.
        """

    def __init__(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        importance: typing.Callable,
        target_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR],
        ch_sparsity: float = 0.5,
        layer_ch_sparsity: typing.Dict[nn.Module, float] = None,
        pruning_steps: int = 1,
        pruning_rate_scheduler: typing.Callable = linear_scheduler,
        max_ch_sparsity: float = 1.0,
        global_pruning: bool = False,
        channel_groups: typing.Dict[nn.Module, int] = dict(),
        ignored_layers: typing.List[nn.Module] = None,
        customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None,
        user_defined_parameters: typing.List[nn.Parameter] = None,
        output_transform: typing.Callable = None,
        round_to: int = None,
    ):
        self.model = model
        self.importance = importance
        self.ch_sparsity = ch_sparsity
        self.layer_ch_sparsity = layer_ch_sparsity if layer_ch_sparsity is not None else {}
        self.target_module_types = target_module_types
        self.channel_groups = channel_groups
        self.global_pruning = global_pruning
        self.max_ch_sparsity = max_ch_sparsity

        # Build dependency graph
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            output_transform=output_transform,
            user_defined_parameters=user_defined_parameters,
            customized_pruners=customized_pruners,
        )
        self.ignored_layers = ignored_layers if ignored_layers is not None else []
        self.pruning_steps = pruning_steps
        self.pruning_rate_scheduler = pruning_rate_scheduler
        self.current_step = 0
        self.round_to = round_to

        # Record input and output channels
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        for m in self.DG.module2node.keys():
            if ops.module2type(m) in self.DG.REGISTERED_PRUNERS:
                self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
                self.layer_init_in_ch[m] = self.DG.get_in_channels(m)


        # The universal channel sparsity
        self.per_step_ch_sparsity = self.pruning_rate_scheduler(
            self.ch_sparsity, self.pruning_steps
        )

        # The customized channel sparsity for different layers
        self.layer_ch_sparsity = {}  
        if layer_ch_sparsity is not None:
            for module in layer_ch_sparsity:
                sparsity = layer_ch_sparsity[module]
                for submodule in m.modules():
                    prunable_types = [ ops.type2class(prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys() ]
                    if isinstance(submodule, prunable_types):
                        self.layer_ch_sparsity[submodule] = self.pruning_rate_scheduler(
                            sparsity, self.pruning_steps
                        )

        if self.global_pruning:
            initial_total_channels = 0
            for group in self.get_all_groups():
                initial_total_channels+=self.DG.get_out_channels(group[0][0].target.module) # utils.count_prunable_out_channels( group[0][0].target.module )
            self.initial_total_channels = initial_total_channels

        # for group convs
        for m in self.model.modules():
            if isinstance(m, ops.TORCH_CONV) \
                and m.groups>1 \
                    and m.groups!=m.out_channels:
                        self.channel_groups[m] = m.groups
                        #print(m, m.weight.shape)

    def get_target_sparsity(self, module):
        s = self.layer_ch_sparsity.get(module, self.per_step_ch_sparsity)[self.current_step]
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
            
            if not isinstance(m, tuple(self.target_module_types)):
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

    def estimate_importance(self, group):
        return self.importance(group)

    def _check_sparsity(self, group):
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler

            if function.is_out_channel_pruner(pruning_fn):
                target_sparsity = self.get_target_sparsity(module)
                layer_out_ch = self.DG.get_out_channels(module)
                if layer_out_ch <= self.layer_init_out_ch[module] * (
                    1 - target_sparsity
                ):
                    return False

            elif function.is_in_channel_pruner(pruning_fn):
                target_sparsity = self.get_target_sparsity(module)
                layer_in_ch = self.DG.get_in_channels(module)
                if layer_in_ch <= self.layer_init_in_ch[module] * (
                    1 - target_sparsity
                ):
                    return False
        return True

    def get_channel_groups(self, group):
        if isinstance(self.channel_groups, int):
            return self.channel_groups
        for dep, _ in group:
            module = dep.target.module
            if module in self.channel_groups:
                return self.channel_groups[module]
        return 1 # no channel grouping

    def prune_local(self):
        if self.current_step == self.pruning_steps:
            return
        for group in self.get_all_groups():

            # check pruning rate
            if self._check_sparsity(group):
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler

                
                imp = self.estimate_importance(group)
                current_channels = self.DG.get_out_channels(module)
                target_sparsity = self.get_target_sparsity(module)
                n_pruned = current_channels - int(
                    self.layer_init_out_ch[module] *
                    (1 - target_sparsity)
                )
                if self.round_to:
                    n_pruned = n_pruned - (n_pruned % self.round_to)
                #print(n_pruned)
                ch_groups = self.get_channel_groups(group)
                imp = imp.view(ch_groups, -1).sum(0)
                imp_argsort = torch.argsort(imp)
                pruning_idxs = imp_argsort[:(n_pruned//ch_groups)]
                offset = current_channels//ch_groups
                if ch_groups>1:
                    pruning_idxs = torch.cat([pruning_idxs+offset*i for i in range(ch_groups)], 0)
                #print(ch_groups, imp.shape, (n_pruned//ch_groups), pruning_idxs.shape)
                #print(pruning_idxs.tolist())
                group = self.DG.get_pruning_group(
                    module, pruning_fn, pruning_idxs.tolist())
                if self.DG.check_pruning_group(group):
                    #print(group)
                    group.exec()
                

    def prune_global(self):
        if self.current_step == self.pruning_steps:
            return
        global_importance = []
        for group in self.get_all_groups():
            imp = self.estimate_importance(group)
            global_importance.append((group, imp))

        imp = torch.cat([local_imp[-1] for local_imp in global_importance], dim=0)
        target_sparsity = self.per_step_ch_sparsity[self.current_step]
        n_pruned = len(imp) - int(
            self.initial_total_channels *
            (1 - target_sparsity)
        )
        if n_pruned<0:
            return
        topk_imp, _ = torch.topk(imp, k=n_pruned, largest=False)
        # global pruning by threshold
        thres = topk_imp[-1]
        for group, imp in global_importance:
            module = group[0][0].target.module
            pruning_fn = group[0][0].handler
            pruning_indices = (imp <= thres).nonzero().view(-1).tolist()
            group = self.DG.get_pruning_group(module, pruning_fn, pruning_indices)
            if self._check_sparsity(group) and self.DG.check_pruning_group(group):
                group.exec()
    