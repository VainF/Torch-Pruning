from .. import dependency, functional, utils
from numbers import Number
from typing import Callable
from .basepruner import BasePruner


class MagnitudeBasedPruner(BasePruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        steps=1,
        scheduler: Callable = None,
        ch_sparsity=0.5,
        layer_ch_sparsity=None,
        round_to=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
    ):
        super(MagnitudeBasedPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            steps=steps,
            scheduler=scheduler,
            ch_sparsity=ch_sparsity,
            layer_ch_sparsity=layer_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            user_defined_parameters=user_defined_parameters,
            output_transform=output_transform,
        )
        self.importance = importance

    def estimate_importance(self, plan):
        return self.importance(plan)
