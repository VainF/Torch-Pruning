import torch.nn as nn
import numpy as np
import torch
from operator import add
from numbers import Number
from collections import namedtuple

UnwrappedParameters = namedtuple('UnwrappedParameters', ['parameters', 'pruning_dim'])
GroupItem = namedtuple('GroupItem', ['dep', 'idxs']) # Group = [GroupItem_1, GroupItem_2, ...]

class PruningIndex(namedtuple("_PruingIndex", ["idx", "root_idx"])):
    def __repr__(self):
        return str( (self.idx, self.root_idx) )

def to_plain_idxs(idxs: PruningIndex):
    if len(idxs)==0 or not isinstance(idxs[0], PruningIndex):
        return idxs
    return [i.idx for i in idxs]

def to_root_idxs(idxs: PruningIndex):
    if len(idxs)==0 or not isinstance(idxs[0], PruningIndex):
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


class _FlattenIndexMapping(object):
    def __init__(self, stride=1, reverse=False):
        self._stride = stride
        self.reverse = reverse

    def __call__(self, idxs: PruningIndex):
        new_idxs = []
        if self.reverse == True:
            for i in idxs:
                new_idxs.append( PruningIndex( idx = (i.idx // self._stride), root_idx=i.root_idx ) )
            new_idxs = list(set(new_idxs))
        else:
            for i in idxs:
                new_idxs.extend(
                    [ PruningIndex(idx=k, root_idx=i.root_idx) for k in range(i.idx * self._stride, (i.idx + 1) * self._stride) ]  
                )
        return new_idxs


class _ConcatIndexMapping(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs: PruningIndex):

        if self.reverse == True:
            new_idxs = [
                PruningIndex(idx = i.idx - self.offset[0], root_idx=i.root_idx )
                for i in idxs
                if (i.idx >= self.offset[0] and i.idx < self.offset[1])
            ]
        else:
            new_idxs = [ PruningIndex(idx=i.idx + self.offset[0], root_idx=i.root_idx) for i in idxs]
        return new_idxs


class _SplitIndexMapping(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs: PruningIndex):
        if self.reverse == True:
            new_idxs = [ PruningIndex(idx=i.idx + self.offset[0], root_idx=i.root_idx) for i in idxs]
        else:
            new_idxs = [
                PruningIndex(idx = i.idx - self.offset[0], root_idx=i.root_idx)
                for i in idxs
                if (i.idx >= self.offset[0] and i.idx < self.offset[1])
            ]
        return new_idxs

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
