from re import S
from .. import importance, dependency, functional, utils
from .scheduler import linear_scheduler
import torch
import torch.nn as nn
import typing


class MetaPruner():
    """
        Basic Pruner for channel pruning.

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
        ch_sparsity: float = 0.5,
        layer_ch_sparsity: typing.Dict[nn.Module, float] = None,
        pruning_steps: int = 1,
        pruning_rate_scheduler: typing.Callable = linear_scheduler,
        max_ch_sparsity: float = 1.0,
        global_pruning: bool = False,
        ignored_layers: typing.List[nn.Module] = None,
        user_defined_parameters: typing.List[nn.Parameter] = None,
        output_transform: typing.Callable = None,
        round_to: int = None,
    ):
        self.model = model
        self.importance = importance
        self.ch_sparsity = ch_sparsity
        self.layer_ch_sparsity = layer_ch_sparsity if layer_ch_sparsity is not None else {}

        self.global_pruning = global_pruning
        self.max_ch_sparsity = max_ch_sparsity

        # Build dependency graph
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            output_transform=output_transform,
            user_defined_parameters=user_defined_parameters,
        )
        self.ignored_layers = ignored_layers if ignored_layers is not None else []
        self.pruning_steps = pruning_steps
        self.pruning_rate_scheduler = pruning_rate_scheduler
        self.current_step = 0
        self.round_to = round_to

        # Record input and output channels
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        for m in self.model.modules():
            if isinstance(m, (dependency.TORCH_CONV, dependency.TORCH_LINEAR)):
                self.layer_init_out_ch[m] = utils.count_prunable_out_channels(m)
                self.layer_init_in_ch[m] = utils.count_prunable_in_channels(m)

        # setup channel sparsity
        self.per_step_ch_sparsity = self.pruning_rate_scheduler(
            self.ch_sparsity, self.pruning_steps
        )
        self.layer_ch_sparsity = {}  # user specified channel sparsity
        if layer_ch_sparsity is not None:
            for m in layer_ch_sparsity:
                s = layer_ch_sparsity[m]
                for submodule in m.modules():
                    if isinstance(submodule, (dependency.TORCH_CONV, dependency.TORCH_LINEAR)):
                        self.layer_ch_sparsity[submodule] = self.pruning_rate_scheduler(
                            s, self.pruning_steps
                        )

    def get_target_sparsity(self, module):
        s = self.layer_ch_sparsity.get(module, self.per_step_ch_sparsity)[self.current_step]
        return min(s, self.max_ch_sparsity)

    def reset(self):
        self.current_step = 0

    def regularize(self, model, loss):
        """ Model regularizor
        """
        pass
    
    def get_all_cliques(self):
        visited_layers = []
        for m in self.model.modules():
            if m in self.ignored_layers:
                continue
            if isinstance(m, dependency.TORCH_CONV):
                pruning_fn = functional.prune_conv_out_channel
            elif isinstance(m, dependency.TORCH_LINEAR):
                pruning_fn = functional.prune_linear_out_channel
            else:
                continue

            if m in visited_layers and pruning_fn in self.DG.out_channel_pruners:
                continue

            layer_channels = utils.count_prunable_out_channels(m)
            clique = self.DG.get_pruning_clique(
                m, pruning_fn, list(range(layer_channels)))
            prunable_clique = True
            for dep, _ in clique:
                module = dep.target.module
                pruning_fn = dep.handler
                if pruning_fn in self.DG.out_channel_pruners:
                    visited_layers.append(module)
                    if module in self.ignored_layers:
                        prunable_clique = False
            if prunable_clique:
                yield clique

    def step(self):
        if self.global_pruning:
            self.prune_global()
        else:
            self.prune_local()
        self.current_step += 1

    def estimate_importance(self, clique):
        return self.importance(clique)

    def _check_sparsity(self, clique):
        for dep, _ in clique:
            module = dep.target.module
            pruning_fn = dep.handler

            if pruning_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                layer_per_step_ch_sparsity = self.get_target_sparsity(module)
                layer_out_ch = utils.count_prunable_out_channels(module)
                if layer_out_ch <= self.layer_init_out_ch[module] * (
                    1 - layer_per_step_ch_sparsity
                ):
                    return False

            elif pruning_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                layer_per_step_ch_sparsity = self.get_target_sparsity(module)
                layer_in_ch = utils.count_prunable_in_channels(module)
                if layer_in_ch <= self.layer_init_in_ch[module] * (
                    1 - layer_per_step_ch_sparsity
                ):
                    return False
        return True

    def prune_local(self):
        if self.current_step == self.pruning_steps:
            return
        for clique in self.get_all_cliques():
            # check pruning rate
            if self._check_sparsity(clique):
                module = clique[0][0].target.module
                pruning_fn = clique[0][0].handler
                imp = self.estimate_importance(clique)
                current_channels = utils.count_prunable_out_channels(module)
                target_sparsity = self.get_target_sparsity(module)
                n_pruned = current_channels - int(
                    self.layer_init_out_ch[module] *
                    (1 - target_sparsity)
                )
                if self.round_to:
                    n_pruned = n_pruned - (n_pruned % self.round_to)
                imp_argsort = torch.argsort(imp)
                pruning_idxs = imp_argsort[:n_pruned].tolist()
                clique = self.DG.get_pruning_clique(
                    module, pruning_fn, pruning_idxs)
                if self.DG.check_pruning_clique(clique):
                    clique.exec()

    def prune_global(self):
        if self.current_step == self.pruning_steps:
            return

        global_importance = []
        for clique in self.get_all_cliques():
            imp = self.estimate_importance(clique)
            global_importance.append((clique, imp))

        # get pruning threshold by ranking
        imp = torch.cat([local_imp[-1]
                        for local_imp in global_importance], dim=0)
        topk_imp, indices = torch.topk(
            imp, k=int(len(imp) * self.ch_sparsity), largest=False
        )

        thres = topk_imp[-1]
        for clique, imp in global_importance:
            module = clique[0][0].target.module
            pruning_fn = clique[0][0].handler
            pruning_indices = (imp < thres).nonzero().view(-1).tolist()
            clique = self.DG.get_pruning_clique(
                module, pruning_fn, pruning_indices)
            if self._check_sparsity(clique) and self.DG.check_pruning_clique(clique):
                clique.exec()
