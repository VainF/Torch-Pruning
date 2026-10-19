"""Torch-Pruning: A framework for structural pruning of PyTorch models.

This package provides tools for structural pruning of deep neural networks,
including dependency graph analysis and various pruning algorithms.
"""

# Core imports
from . import _helpers, utils
from .dependency import *
from .pruner import *
from .pruner import importance
from .serialization import save, load, state_dict, load_state_dict

__version__ = "1.6.0"
__author__ = "Gongfan Fang"
__email__ = "gongfan@u.nus.edu"