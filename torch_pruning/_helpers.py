import torch.nn as nn
import numpy as np
import torch
from operator import add
from numbers import Number
from collections import namedtuple

UnwrappedParameters = namedtuple('UnwrappedParameters', ['parameters', 'pruning_dim'])

def normalize_saved_dim(dim, ndim=None):
    """Turn a dim saved by autograd (e.g. ``grad_fn._saved_dim``) into a signed, and if ``ndim`` is known non-negative, dim.

    Recent PyTorch releases expose a negative dim, e.g. the one of ``torch.cat(xs, dim=-1)`` or
    ``torch.split(x, n, dim=-1)``, as an unsigned 64-bit integer (``-1 -> 2**64 - 1``), while older
    releases expose ``-1``. Both are mapped to ``ndim - 1`` here.
    """
    if dim is None:
        return None
    dim = int(dim)
    if dim >= 2**63:
        dim -= 2**64
    if ndim is not None and -ndim <= dim < 0:
        dim += ndim
    return dim

def grad_fn_output_shape(grad_fn, output_nr=0):
    """Shape of the ``output_nr``-th forward output of the op that created ``grad_fn``, or None if unknown."""
    try:
        return tuple(grad_fn._input_metadata[output_nr].shape)
    except Exception: # _input_metadata is a private attribute and is missing in old PyTorch releases
        return None

def grad_fn_num_outputs(grad_fn):
    """Number of forward outputs of the op that created ``grad_fn`` (e.g. the number of chunks of torch.unbind), or None if unknown."""
    try:
        return len(grad_fn._input_metadata)
    except Exception:
        return None

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
