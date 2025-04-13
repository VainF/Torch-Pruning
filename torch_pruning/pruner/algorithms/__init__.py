from .base_pruner import BasePruner

# Regularization-based pruner
from .batchnorm_scale_pruner import BNScalePruner
from .group_norm_pruner import GroupNormPruner
from .growing_reg_pruner import GrowingRegPruner

# deprecated
from .compatibility import MetaPruner, MagnitudePruner