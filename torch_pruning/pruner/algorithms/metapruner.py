import torch
import torch.nn as nn
import typing

from .scheduler import linear_scheduler
from ..import function
from ... import ops, dependency


class MetaPruner:
    """
        Meta pruner for structural pruning. 

        Args:

            # Basic
            * model (nn.Module): A to-be-pruned model
            * example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
            * importance (Callable): importance estimator. 
            * global_pruning (bool): enable global pruning. Default: False.
            * ch_sparsity (float): global channel sparisty. Default: 0.5.
            * ch_sparsity_dict (Dict[nn.Module, float]): layer-specific sparsity. Will cover ch_sparsity if specified. Default: None.
            * max_ch_sparsity (float): maximum channel sparsity. Default: 1.0.
            * iterative_steps (int): number of steps for iterative pruning. Default: 1.
            * iterative_sparsity_scheduler (Callable): scheduler for iterative pruning. Default: linear_scheduler.
            * ignored_layers (List[nn.Module | typing.Type]): ignored modules. Default: None.
            * round_to (int): channel rounding. E.g., round_to=8 means channels will be rounded to 8x. Default: None.
            
            # Adavanced
            * customized_pruners (dict): a dict containing module-pruner pairs. Default: None.
            * unwrapped_parameters (dict): a dict containing unwrapped parameters & pruning dims. Default: None.
            * root_module_types (list): types of prunable modules. Default: [nn.Conv2d, nn.Linear, nn.LSTM].
            * forward_fn (Callable): A function to execute model.forward. Default: None.
            * output_transform (Callable): A function to transform network outputs. Default: None.
        """

    def __init__(
        self,
        # Basic
        model: nn.Module, # a simple pytorch model
        example_inputs: torch.Tensor, # a dummy input for graph tracing. Should be on the same 
        importance: typing.Callable, # tp.importance.Importance for group importance estimation
        global_pruning: bool = False, # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        ch_sparsity: float = 0.5,  # channel/dim sparsity
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None, # layer-specific sparsity, will cover ch_sparsity if specified
        max_ch_sparsity: float = 1.0, # maximum sparsity. useful if over-pruning happens.
        iterative_steps: int = 1,  # for iterative pruning
        iterative_sparsity_scheduler: typing.Callable = linear_scheduler, # scheduler for iterative pruning.
        ignored_layers: typing.List[nn.Module] = None, # ignored layers
        round_to: int = None,  # round channels to a multiple of round_to

        # Advanced
        channel_groups: typing.Dict[nn.Module, int] = dict(), # channel groups for layers like group convs & group norms
        customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None, # pruners for customized layers. E.g., {nn.Linear: my_linear_pruner}
        unwrapped_parameters: typing.Dict[nn.Parameter, int] = None, # unwrapped nn.Parameters & pruning_dims. For example, {ViT.pos_emb: 0}
        root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        forward_fn: typing.Callable = None, # a function to execute model.forward
        output_transform: typing.Callable = None, # a function to transform network outputs
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
            forward_fn=forward_fn,
            output_transform=output_transform,
            unwrapped_parameters=unwrapped_parameters,
            customized_pruners=customized_pruners,
        )

        # Ignored layers
        self.ignored_layers = []
        if ignored_layers:
            for layer in ignored_layers:
                self.ignored_layers.extend(list(layer.modules()))

        # Iterative pruning
        # The pruner will prune the model iteratively for several steps to achieve the target sparsity
        # E.g., if iterative_steps=5, ch_sparsity=0.5, the sparsity of each step will be [0.1, 0.2, 0.3, 0.4, 0.5]
        self.iterative_steps = iterative_steps
        self.iterative_sparsity_scheduler = iterative_sparsity_scheduler
        self.current_step = 0

        # initial channels/dims for each layer
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        for m in self.DG.module2node.keys():
            if ops.module2type(m) in self.DG.REGISTERED_PRUNERS:
                self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
                self.layer_init_in_ch[m] = self.DG.get_in_channels(m)

        # channel sparsity for each iterative step
        self.per_step_ch_sparsity = self.iterative_sparsity_scheduler(
            self.ch_sparsity, self.iterative_steps
        )

        # The layer-specific sparsity will cover the global sparsity if specified
        self.ch_sparsity_dict = {}
        if ch_sparsity_dict is not None:
            for module in ch_sparsity_dict:
                sparsity = ch_sparsity_dict[module]
                for submodule in module.modules():
                    prunable_types = tuple([ops.type2class(
                        prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys()])
                    if isinstance(submodule, prunable_types):
                        self.ch_sparsity_dict[submodule] = self.iterative_sparsity_scheduler(
                            sparsity, self.iterative_steps
                        )
        
        # detect group convs & group norms
        for m in self.model.modules():
            if isinstance(m, ops.TORCH_CONV) \
                and m.groups > 1 \
                    and m.groups != m.out_channels:
                self.channel_groups[m] = m.groups
            if isinstance(m, ops.TORCH_GROUPNORM):
                self.channel_groups[m] = m.num_groups
        
        # count the number of total channels at initialization
        if self.global_pruning:
            initial_total_channels = 0
            for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types):
                ch_groups = self.get_channel_groups(group)
                # utils.count_prunable_out_channels( group[0][0].target.module )
                initial_total_channels += (self.DG.get_out_channels(
                    group[0][0].target.module) // ch_groups)
            self.initial_total_channels = initial_total_channels
    
    def pruning_history(self) -> typing.List[typing.Tuple[str, bool, typing.Union[list, tuple]]]:
        return self.DG.pruning_history()

    def load_pruning_history(self, pruning_history) -> None:
        self.DG.load_pruning_history(pruning_history)

    def get_target_sparsity(self, module) -> float:
        s = self.ch_sparsity_dict.get(module, self.per_step_ch_sparsity)[self.current_step]
        return min(s, self.max_ch_sparsity)

    def reset(self) -> None:
        self.current_step = 0

    def regularize(self, model, loss) -> typing.Any:
        """ Model regularizor
        """
        pass

    def step(self, interactive=False)-> typing.Union[typing.Generator, None]:
        self.current_step += 1
        pruning_fn = self.prune_global if self.global_pruning else self.prune_local
        if interactive: # yield groups for interactive pruning
            return pruning_fn() 
        else:
            for group in pruning_fn():
                group.prune()

    def estimate_importance(self, group, ch_groups=1) -> torch.Tensor:
        return self.importance(group, ch_groups=ch_groups)

    def _check_sparsity(self, group) -> bool:
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if dep.target.type == ops.OPTYPE.PARAMETER:
                continue
            if self.DG.is_out_channel_pruning_fn(pruning_fn):
                target_sparsity = self.get_target_sparsity(module)
                layer_out_ch = self.DG.get_out_channels(module)
                if layer_out_ch is None: continue
                if layer_out_ch < self.layer_init_out_ch[module] * (
                    1 - self.max_ch_sparsity
                ) or layer_out_ch == 1:
                    return False

            elif self.DG.is_in_channel_pruning_fn(pruning_fn):
                layer_in_ch = self.DG.get_in_channels(module)
                if layer_in_ch is None: continue
                if layer_in_ch < self.layer_init_in_ch[module] * (
                    1 - self.max_ch_sparsity
                ) or layer_in_ch == 1:
                    return False
        return True

    def get_channel_groups(self, group) -> int:
        if isinstance(self.channel_groups, int):
            return self.channel_groups
        for dep, _ in group:
            module = dep.target.module
            if module in self.channel_groups:
                return self.channel_groups[module]
        return 1  # no channel grouping

    def prune_local(self) -> typing.Generator:
        if self.current_step > self.iterative_steps:
            return
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types):
            # check pruning rate
            if self._check_sparsity(group):
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler

                ch_groups = self.get_channel_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups)
                if imp is None: continue
                current_channels = self.DG.get_out_channels(module)
                target_sparsity = self.get_target_sparsity(module)
                n_pruned = current_channels - int(
                    self.layer_init_out_ch[module] *
                    (1 - target_sparsity)
                )

                if self.round_to:
                    n_pruned = n_pruned - (n_pruned % self.round_to)
    
                if n_pruned <= 0:
                    continue
                if ch_groups > 1:
                    imp = imp[:len(imp)//ch_groups]
                imp_argsort = torch.argsort(imp)
                pruning_idxs = imp_argsort[:(n_pruned//ch_groups)]
                if ch_groups > 1:
                    group_size = current_channels//ch_groups
                    pruning_idxs = torch.cat(
                        [pruning_idxs+group_size*i for i in range(ch_groups)], 0)
                group = self.DG.get_pruning_group(
                    module, pruning_fn, pruning_idxs.tolist())
                if self.DG.check_pruning_group(group):
                    yield group

    def prune_global(self) -> typing.Generator:
        if self.current_step > self.iterative_steps:
            return
        global_importance = []
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types):
            if self._check_sparsity(group):
                ch_groups = self.get_channel_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups)
                if imp is None: continue
                if ch_groups > 1:
                    imp = imp[:len(imp)//ch_groups]
                global_importance.append((group, ch_groups, imp))

        if len(global_importance) == 0:
            return
                
        imp = torch.cat([local_imp[-1]
                        for local_imp in global_importance], dim=0)
        target_sparsity = self.per_step_ch_sparsity[self.current_step]
        n_pruned = len(imp) - int(
            self.initial_total_channels *
            (1 - target_sparsity)
        )
        if n_pruned <= 0:
            return
        topk_imp, _ = torch.topk(imp, k=n_pruned, largest=False)
        
        # global pruning through thresholding
        thres = topk_imp[-1]
        for group, ch_groups, imp in global_importance:
            module = group[0][0].target.module
            pruning_fn = group[0][0].handler
            pruning_indices = (imp <= thres).nonzero().view(-1)
            if ch_groups > 1:
                group_size = self.DG.get_out_channels(module)//ch_groups
                pruning_indices = torch.cat(
                    [pruning_indices+group_size*i for i in range(ch_groups)], 0)
            if self.round_to:
                n_pruned = len(pruning_indices)
                n_pruned = n_pruned - (n_pruned % self.round_to)
                pruning_indices = pruning_indices[:n_pruned]
            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices.tolist())
            if self.DG.check_pruning_group(group):
                yield group
