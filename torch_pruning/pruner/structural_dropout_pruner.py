from .. import dependency, functional, utils
from numbers import Number
from typing import Callable
from .basepruner import LocalPruner
import torch
import torch.nn as nn

def imp_to_prob(x, scale=1.0):
    return torch.nn.functional.sigmoid( (x - x.mean()) / (x.std() + 1e-8) * scale )  #torch.clamp( (x - x.min()) / (x.max() - x.min()), min=0.4)

class StructrualDropout(nn.Module):
    def __init__(self, p):
        super(StructrualDropout, self).__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        C = x.shape[1]
        if self.mask is None:
            self.mask = (torch.cuda.FloatTensor(C, device=x.device).uniform_() > self.p).view(1, -1, 1, 1)
        res = x * self.mask
        return res
    
    def reset(self, p):
        self.p = p
        self.mask = None

class StructrualDropoutPruner(LocalPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        total_steps=1,
        p=0.1,
        pruning_rate_scheduler: Callable = None,
        ch_sparsity=0.5,
        layer_ch_sparsity=None,
        round_to=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
    ):
        super(StructrualDropoutPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            total_steps=total_steps,
            pruning_rate_scheduler=pruning_rate_scheduler,
            ch_sparsity=ch_sparsity,
            layer_ch_sparsity=layer_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            user_defined_parameters=user_defined_parameters,
            output_transform=output_transform,
        )
        self.importance = importance
        self.module2dropout = {}
        self.p = p
        self.plans = self.get_all_plans()

    def estimate_importance(self, plan):
        return self.importance(plan)

    def structrual_dropout(self, module, input, output):
        return self.module2dropout[module][0](output)

    def regularize(self, model):
        pass 
        #for plan in self.plans:
        #    module = plan[0][0].target.module
        #    imp = self.estimate_importance(plan)
        #    imp2prob = imp_to_prob(imp)
        #    dropout_layer = self.module2dropout[module][0]
        #    dropout_layer.reset(p=imp2prob)
    
    def register_structural_dropout(self, module):
        for plan in self.plans:
            imp = self.estimate_importance(plan)
            dropout_layer = StructrualDropout(p=self.p)
            for dep, _ in plan:
                module = dep.target.module
                if self.ignored_layers is not None and module in self.ignored_layers:
                    continue
                if module in self.module2dropout:
                    continue
                if dep.handler not in self.DG.out_channel_pruners:
                    continue
                hook = module.register_forward_hook(self.structrual_dropout)                
                self.module2dropout[module] = (dropout_layer, hook)
    
    def remove_structural_dropout(self):
        for m, (_, hook) in self.module2dropout.items():
            hook.remove()

        
        
            

        