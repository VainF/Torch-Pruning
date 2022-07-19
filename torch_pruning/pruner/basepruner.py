from .. import importance, dependency, functional, utils
from numbers import Number
from typing import Callable
import abc
import torch

def linear_scheduler(layer_ch_sparsity, steps):
    return [((i + 1) / float(steps)) * layer_ch_sparsity for i in range(steps)]

class MetaPruner(abc.ABC):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        total_steps = 1,
        pruning_rate_scheduler=linear_scheduler,
        ch_sparsity=0.5,
        layer_ch_sparsity=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
        round_to=None,
    ):
        self.model = model
        self.importance = importance
        self.ch_sparsity = ch_sparsity
        self.round_to = round_to
        self.layer_ch_sparsity = layer_ch_sparsity if layer_ch_sparsity is not None else {}
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
        
        for m in self.model.modules(): #self.model.modules():
            if isinstance(m, (dependency.TORCH_CONV, dependency.TORCH_LINEAR)):
                self.layer_init_out_ch[m] = utils.count_prunable_out_channels(m)
                self.layer_init_in_ch[m] = utils.count_prunable_in_channels(m)
                self.per_step_ch_sparsity[m] = self.pruning_rate_scheduler(
                    self.ch_sparsity, self.total_steps
                )
        # override
        for m in self.model.modules(): 
            if m in self.layer_ch_sparsity:
                sublayer_ch_sparsity = self.layer_ch_sparsity[m]
                for mi in m.modules():
                    if isinstance(mi, (dependency.TORCH_CONV, dependency.TORCH_LINEAR)):
                        self.per_step_ch_sparsity[mi] = self.pruning_rate_scheduler(
                            sublayer_ch_sparsity, self.total_steps
                        )

    def reset(self):
        self.current_step = 0

    def regularize(self, model):
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
            plan = self.DG.get_pruning_plan(
                m, pruning_fn, list(range(layer_channels))
            )
            active_plan = True
            for dep, _ in plan:
                module = dep.target.module
                pruning_fn = dep.handler
                if pruning_fn in self.DG.out_channel_pruners:
                    visited_layers.append(module)
                    if module in self.ignored_layers:
                        active_plan=False
            if active_plan:
                yield plan

    @abc.abstractclassmethod
    def step(self):
        pass
        
    def estimate_importance(self, plan):
        return self.importance(plan)


class LocalPruner(MetaPruner):

    def step(self):
        if self.current_step == self.total_steps:
            return
        for plan in self.get_all_plans():
            # check pruning rate
            if self._is_valid(plan):
                module = plan[0][0].target.module
                pruning_fn = plan[0][0].handler
                imp = self.estimate_importance(plan)
                current_channels = utils.count_prunable_out_channels(module)
                layer_step_ch_sparsity = self.per_step_ch_sparsity[module][self.current_step]
                n_pruned = current_channels - int(
                    self.layer_init_out_ch[module] * (1 - layer_step_ch_sparsity)
                )
                if self.round_to:
                    n_pruned = n_pruned % self.round_to * self.round_to
                imp_argsort = torch.argsort(imp)
                pruning_idxs = imp_argsort[:n_pruned].tolist()
                plan = self.DG.get_pruning_plan(module, pruning_fn, pruning_idxs)
                if self.DG.check_pruning_plan(plan):
                    plan.exec()
        self.current_step += 1

    def _is_valid(self, plan):
        for dep, _ in plan:
            if dep.target.module in self.per_step_ch_sparsity :
                if dep.handler in [
                    functional.prune_conv_out_channel,
                    functional.prune_linear_out_channel,
                ]:
                    layer_step_ch_sparsity = self.per_step_ch_sparsity[dep.target.module][self.current_step]
                    layer_channels = utils.count_prunable_out_channels(dep.target.module)
                    if layer_channels <= self.layer_init_out_ch[dep.target.module] * (1 - layer_step_ch_sparsity):
                        return False
                elif dep.handler in [
                    functional.prune_conv_in_channel,
                    functional.prune_linear_in_channel,
                ]:
                    layer_step_ch_sparsity = self.per_step_ch_sparsity[dep.target.module][self.current_step]
                    layer_channels = utils.count_prunable_in_channels(dep.target.module)
                    if layer_channels <= self.layer_init_in_ch[dep.target.module] * (1 - layer_step_ch_sparsity):
                        return False
        return True




class GlobalPruner(MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        total_steps=1,
        pruning_rate_scheduler=linear_scheduler,
        ch_sparsity=0.5,
        max_ch_sparsity=1.0,
        layer_ch_sparsity=None,
        round_to=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
    ):  
        super(GlobalPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            total_steps=total_steps,
            pruning_rate_scheduler = pruning_rate_scheduler,
            ch_sparsity=ch_sparsity,
            layer_ch_sparsity=layer_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            user_defined_parameters=user_defined_parameters,
            output_transform=output_transform,
        )

        self.max_ch_sparsity = max_ch_sparsity
        # global channel sparsity shared by all layers
        self.per_step_ch_sparsity = self.pruning_rate_scheduler(
            self.ch_sparsity, self.total_steps
        )
        self.current_step = 0

    def reset(self):
        self.current_step = 0

    def regularize(self, model):
        pass
    
    def step(self):
        if self.current_step == self.total_steps:
            return
        
        global_importance = []
        for plan in self.get_all_plans():
            imp = self.estimate_importance(plan)
            global_importance.append((plan, imp))
        
        # get pruning threshold by ranking
        imp = torch.cat([local_imp[-1] for local_imp in global_importance], dim=0)
        topk_imp, indices = torch.topk(imp, k=int(len(imp) * self.ch_sparsity))
        thres = topk_imp[-1]

        for plan, imp in global_importance:
            module = plan[0][0].target.module
            pruning_fn = plan[0][0].handler
            pruning_indices = (imp<thres).nonzero().view(-1).tolist()
            plan = self.DG.get_pruning_plan(module, pruning_fn, pruning_indices)
            n_prune = self._adjust_sparsity(plan)
            if n_prune<len(pruning_indices):
                pruning_indices = pruning_indices[:n_prune]
                plan = self.DG.get_pruning_plan(module, pruning_fn, pruning_indices)      
            if self.DG.check_pruning_plan(plan):
                plan.exec()
        self.current_step += 1

    def _adjust_sparsity(self, plan):
        new_idxs = plan[0][1]
        n_prune = len(new_idxs)
        for i, (dep, idxs) in enumerate(plan):
            module = dep.target.module
            if dep.handler in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                max_ch_sparsity = self.layer_ch_sparsity.get(module, self.max_ch_sparsity)
                min_layer_channels = self.layer_init_out_ch[module] * (1 - max_ch_sparsity)
                layer_channels = utils.count_prunable_out_channels(module)
                if len(idxs) <= int(layer_channels - min_layer_channels):
                    continue
                else:
                    n_prune = int(layer_channels - min_layer_channels)
            elif dep.handler in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                max_ch_sparsity = self.layer_ch_sparsity.get(module, self.max_ch_sparsity)
                min_layer_channels = self.layer_init_in_ch[module] * (1 - max_ch_sparsity)
                layer_channels = utils.count_prunable_in_channels(module)
                if len(idxs) <= int(layer_channels - min_layer_channels):
                    continue
                else:
                    n_prune = int(layer_channels - min_layer_channels)
        return n_prune
