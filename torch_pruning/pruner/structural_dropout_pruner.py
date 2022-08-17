from .. import dependency, functional, utils
from numbers import Number
from typing import Callable
from .basepruner import MetaPruner
from .scheduler import linear_scheduler
import torch
import torch.nn as nn
import random

class StructrualDropout(nn.Module):
    def __init__(self, p):
        super(StructrualDropout, self).__init__()
        self.p = p
        self.dropout_idxs = None
        self.ch_score = None
        self.cnt = None
        self.module_mapping = {}

    def add_module(self, module, dropout_range):
        self.module_mapping[module] = dropout_range

    def forward(self, x, dropout_range):
        self.C = C = x.shape[1]
        if self.dropout_idxs is None:
            size = dropout_range[1] - dropout_range[0] + 1
            self.dropout_idxs = random.sample(range(size), k=int((1 - self.p) * size))
        idxs = [dropout_range[0] + i for i in self.dropout_idxs]
        mask = torch.zeros(C, device=x.device).view(1, -1, 1, 1)
        mask[:, idxs] = 1
        return x * mask

    @torch.no_grad()
    def step(self, loss=None):
        if loss is not None:
            if self.ch_score is None:
                self.ch_score = torch.zeros(self.C, device=loss.device)
                self.cnt = torch.zeros(self.C, device=loss.device)
            self.ch_score[self.dropout_idxs] += loss
            self.cnt[self.dropout_idxs] += 1
        self.dropout_idxs = None


class StructrualDropoutPruner(MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        pruning_steps=1,
        p=0.3,
        pruning_rate_scheduler: Callable = linear_scheduler,
        ch_sparsity=0.5,
        layer_ch_sparsity=None,
        global_pruning=False,
        max_ch_sparsity=1.0,
        round_to=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
    ):
        super(StructrualDropoutPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            pruning_steps=pruning_steps,
            pruning_rate_scheduler=pruning_rate_scheduler,
            ch_sparsity=ch_sparsity,
            layer_ch_sparsity=layer_ch_sparsity,
            global_pruning=global_pruning,
            max_ch_sparsity=max_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            user_defined_parameters=user_defined_parameters,
            output_transform=output_transform,
        )
        self.importance = importance
        self.module2dropout = {}
        self.p = p
        self.plans = list(self.get_all_cliques())

    def structrual_dropout(self, module, input, output):
        dropout = self.module2dropout[module][0]
        dropout_range = dropout.module_mapping[module]
        return dropout(output, dropout_range)

    def regularize(self, model, loss):
        for clique in self.plans:
            module = clique[0][0].target.module
            self.module2dropout[module][0].step(loss)

    def estimate_importance(self, clique):
        module = clique[0][0].target.module
        dropout = self.module2dropout[module][0]
        return -dropout.ch_score / dropout.cnt

    def register_structural_dropout(self, module):
        for clique in self.plans:
            dropout_layer = StructrualDropout(p=self.ch_sparsity)
            for dep, idxs in clique:
                module = dep.target.module
                if self.ignored_layers is not None and module in self.ignored_layers:
                    continue
                if dep.handler not in self.DG.out_channel_pruners:
                    continue
                hook = module.register_forward_hook(self.structrual_dropout)
                dropout_layer.add_module(module, [min(idxs), max(idxs)])
                self.module2dropout[module] = (dropout_layer, hook)

    def remove_structural_dropout(self):
        for m, (_, hook) in self.module2dropout.items():
            hook.remove()
