from .. import importance, dependency, functional, utils
from numbers import Number
from typing import Callable
import abc
import torch


def linear_scheduler(layer_ch_sparsity, steps):
    return [((i + 1) / float(steps)) * layer_ch_sparsity for i in range(steps)]


class BasePruner:
    def __init__(
        self,
        model,
        example_inputs,
        steps=1,
        scheduler: Callable = None,
        ch_sparsity=0.5,
        layer_ch_sparsity=None,
        round_to=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
    ):
        self.ch_sparsity = ch_sparsity
        self.round_to = round_to
        if layer_ch_sparsity is None:
            layer_ch_sparsity = {}
        self.layer_ch_sparsity = layer_ch_sparsity
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            output_transform=output_transform,
            user_defined_parameters=user_defined_parameters,
        )
        self.model = model
        self.ignored_layers = ignored_layers
        self.steps = steps
        if scheduler is None:
            scheduler = linear_scheduler
        self.scheduler = scheduler
        self.layer_init_ch = {}
        self.per_step_ch_sparsity = {}
        for m in self.model.modules():
            if isinstance(m, (dependency.TORCH_CONV, dependency.TORCH_LINEAR)):
                self.layer_init_ch[m] = utils.count_prunable_channels(m)
                layer_ch_sparsity = self.layer_ch_sparsity.get(m, self.ch_sparsity)
                self.per_step_ch_sparsity[m] = self.scheduler(
                    layer_ch_sparsity, self.steps
                )
        self.current_step = 0

    def reset(self):
        self.current_step = 0

    def step(self):
        if self.current_step == self.steps:
            return

        for m in self.model.modules():
            if m not in self.DG.PRUNABLE_MODULES:
                continue

            if self.ignored_layers is not None and m in self.ignored_layers:
                continue

            if isinstance(m, dependency.TORCH_CONV):
                pruning_fn = functional.prune_conv_out_channel
            elif isinstance(m, dependency.TORCH_LINEAR):
                pruning_fn = functional.prune_linear_out_channel
            else:
                continue

            # check ch_sparsity
            layer_step_ch_sparsity = self.per_step_ch_sparsity[m][self.current_step]
            layer_channels = utils.count_prunable_channels(m)
            full_plan = self.DG.get_pruning_plan(
                m, pruning_fn, list(range(layer_channels))
            )
            for dep, _ in full_plan:
                if dep.target.module in self.layer_ch_sparsity and dep.handler in [
                    functional.prune_conv_out_channel,
                    functional.prune_linear_out_channel,
                ]:
                    layer_step_ch_sparsity = self.per_step_ch_sparsity[
                        dep.target.module
                    ][self.current_step]
                    break

            if layer_channels <= self.layer_init_ch[m] * (1 - layer_step_ch_sparsity):
                continue

            imp = self.estimate_importance(full_plan)
            n_pruned = layer_channels - int(
                self.layer_init_ch[m] * (1 - layer_step_ch_sparsity)
            )
            if self.round_to:
                n_pruned = n_pruned % self.round_to * self.round_to
            imp_argsort = torch.argsort(imp)
            pruning_idxs = imp_argsort[:n_pruned].tolist()

            plan = self.DG.get_pruning_plan(m, pruning_fn, pruning_idxs)
            # print(plan)
            if self.DG.check_pruning_plan(plan):
                plan.exec()
        self.current_step += 1

    @abc.abstractclassmethod
    def estimate_importance(self, plan):
        pass
