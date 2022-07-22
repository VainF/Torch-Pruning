from .. import importance, dependency, functional, utils
from numbers import Number
from typing import Callable
import abc
import torch


def linear_scheduler(layer_ch_sparsity, steps):
    return [((i + 1) / float(steps)) * layer_ch_sparsity for i in range(steps)]


class MetaPruner():
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        total_steps=1,
        pruning_rate_scheduler=linear_scheduler,
        ch_sparsity=0.5,
        layer_ch_sparsity=None,
        global_max_ch_sparsity=1.0,
        global_pruning=False,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
        round_to=None,
    ):
        self.model = model
        self.importance = importance
        self.ch_sparsity = ch_sparsity
        self.round_to = round_to
        self.layer_ch_sparsity = (
            layer_ch_sparsity if layer_ch_sparsity is not None else {}
        )
        self.global_pruning = global_pruning
        self.global_max_ch_sparsity = global_max_ch_sparsity
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            output_transform=output_transform,
            user_defined_parameters=user_defined_parameters,
        )
        if ignored_layers is None:
            ignored_layers = []
        self.ignored_layers = ignored_layers
        self.total_steps = total_steps
        self.current_step = 0

        self.pruning_rate_scheduler = pruning_rate_scheduler
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        self.per_step_ch_sparsity = {}

        for m in self.model.modules():  # self.model.modules():
            if isinstance(m, (dependency.TORCH_CONV, dependency.TORCH_LINEAR)):
                self.layer_init_out_ch[m] = utils.count_prunable_out_channels(m)
                self.layer_init_in_ch[m] = utils.count_prunable_in_channels(m)

        self.per_step_ch_sparsity = self.pruning_rate_scheduler(
            self.ch_sparsity, self.total_steps
        )
        self.layer_ch_sparsity = {} # user specified channel sparsity
        for m in layer_ch_sparsity:
            sublayer_ch_sparsity = layer_ch_sparsity[m]
            for subm in m.modules():
                if isinstance(subm, (dependency.TORCH_CONV, dependency.TORCH_LINEAR)):
                    self.layer_ch_sparsity[subm] = self.pruning_rate_scheduler(
                        sublayer_ch_sparsity, self.total_steps
                    )

    def get_step_ch_sparsity(self, module):
        if self.global_pruning:
            return min( self.layer_ch_sparsity.get(module, self.per_step_ch_sparsity)[self.current_step], self.global_max_ch_sparsity )
        else:
            return self.layer_ch_sparsity.get(module, self.per_step_ch_sparsity)[self.current_step]

    def reset(self):
        self.current_step = 0

    def regularize(self, model, loss):
        pass

    def get_all_plans(self):
        plans = []
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
            plan = self.DG.get_pruning_plan(m, pruning_fn, list(range(layer_channels)))
            active_plan = True
            for dep, _ in plan:
                module = dep.target.module
                pruning_fn = dep.handler
                if pruning_fn in self.DG.out_channel_pruners:
                    visited_layers.append(module)
                    if module in self.ignored_layers:
                        active_plan = False
            if active_plan:
                yield plan

    def step(self):
        if self.global_pruning:
            self.prune_global()
        else:
            self.prune_local()
        self.current_step+=1

    def estimate_importance(self, plan):
        return self.importance(plan)
    
    def _check_sparsity(self, plan):
        for dep, _ in plan:
            module = dep.target.module
            pruning_fn = dep.handler

            if pruning_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                layer_per_step_ch_sparsity = self.get_step_ch_sparsity(module)
                layer_out_ch = utils.count_prunable_out_channels(module)
                if layer_out_ch <= self.layer_init_out_ch[module] * (
                    1 - layer_per_step_ch_sparsity
                ):
                    return False

            elif pruning_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                layer_per_step_ch_sparsity = self.get_step_ch_sparsity(module)
                layer_in_ch = utils.count_prunable_in_channels(module)
                if layer_in_ch <= self.layer_init_in_ch[module] * (
                    1 - layer_per_step_ch_sparsity
                ):
                    return False
        return True

    def prune_local(self):
        if self.current_step == self.total_steps:
            return
        for plan in self.get_all_plans():
            # check pruning rate
            if self._check_sparsity(plan):
                module = plan[0][0].target.module
                pruning_fn = plan[0][0].handler
                imp = self.estimate_importance(plan)
                current_channels = utils.count_prunable_out_channels(module)
                layer_per_step_ch_sparsity = self.get_step_ch_sparsity(module)
                n_pruned = current_channels - int(
                    self.layer_init_out_ch[module] * (1 - layer_per_step_ch_sparsity)
                )
                if self.round_to:
                    n_pruned = n_pruned % self.round_to * self.round_to
                imp_argsort = torch.argsort(imp)
                pruning_idxs = imp_argsort[:n_pruned].tolist()
                plan = self.DG.get_pruning_plan(module, pruning_fn, pruning_idxs)
                if self.DG.check_pruning_plan(plan):
                    plan.exec()

    def prune_global(self):
        if self.current_step == self.total_steps:
            return

        global_importance = []
        for plan in self.get_all_plans():
            imp = self.estimate_importance(plan)
            global_importance.append((plan, imp))

        # get pruning threshold by ranking
        imp = torch.cat([local_imp[-1] for local_imp in global_importance], dim=0)
        topk_imp, indices = torch.topk(
            imp, k=int(len(imp) * self.ch_sparsity), largest=False
        )

        thres = topk_imp[-1]
        for plan, imp in global_importance:
            module = plan[0][0].target.module
            pruning_fn = plan[0][0].handler
            pruning_indices = (imp < thres).nonzero().view(-1).tolist()
            plan = self.DG.get_pruning_plan(module, pruning_fn, pruning_indices)
            if self._check_sparsity(plan) and self.DG.check_pruning_plan(plan):
                plan.exec()

