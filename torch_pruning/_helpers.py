import torch.nn as nn
import numpy as np
import torch
from operator import add
from numbers import Number
from collections import namedtuple

UnwrappedParameters = namedtuple('UnwrappedParameters', ['parameters', 'pruning_dim'])

class GroupItem(namedtuple('_GroupItem', ['dep', 'idxs'])):
    def __new__(cls, dep, idxs):
        """ A tuple of (dep, idxs) where dep is the dependency of the group, and idxs is the list of indices in the group."""
        cls.root_idxs = None # a placeholder. Will be filled by DepGraph
        return super(GroupItem, cls).__new__(cls, dep, idxs)
    
    def __repr__(self):
        return str( (self.dep, self.idxs) )

class _HybridIndex(namedtuple("_PruingIndex", ["idx", "root_idx"])):
    """ A tuple of (idx, root_idx) where idx is the index of the pruned dimension in the current layer, 
    and root_idx is the index of the pruned dimension in the root layer.
    """
    def __repr__(self):
        return str( (self.idx, self.root_idx) )

def to_plain_idxs(idxs: _HybridIndex):
    if len(idxs)==0 or not isinstance(idxs[0], _HybridIndex):
        return idxs
    return [i.idx for i in idxs]

def to_root_idxs(idxs: _HybridIndex):
    if len(idxs)==0 or not isinstance(idxs[0], _HybridIndex):
        return idxs
    return [i.root_idx for i in idxs]

def is_scalar(x):
    if isinstance(x, torch.Tensor):
        return len(x.shape) == 0
    elif isinstance(x, Number):
        return True
    elif isinstance(x, (list, tuple)):
        return False
    return False

class ScalarSum:
    def __init__(self):
        self._results = {}

    def update(self, metric_name, metric_value):
        if metric_name not in self._results:
            self._results[metric_name] = 0
        self._results[metric_name] += metric_value

    def results(self):
        return self._results

    def reset(self):
        self._results = {}


class VectorSum:
    def __init__(self):
        self._results = {}

    def update(self, metric_name, metric_value):
        if metric_name not in self._results:
            self._results[metric_name] = metric_value
        if isinstance(metric_value, torch.Tensor):
            self._results[metric_name] += metric_value
        elif isinstance(metric_value, list):
            self._results[metric_name] = list(
                map(add, self._results[metric_name], metric_value)
            )

    def results(self):
        return self._results

    def reset(self):
        self._results = {}
