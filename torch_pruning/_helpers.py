import torch.nn as nn
import numpy as np
import torch
from operator import add
from numbers import Number


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

    def __call__(self, idxs):
        new_idxs = []
        if self.reverse == True:
            for i in idxs:
                new_idxs.append(i // self._stride)
                new_idxs = list(set(new_idxs))
        else:
            for i in idxs:
                new_idxs.extend(
                    list(range(i * self._stride, (i + 1) * self._stride)))
        return new_idxs


class _ConcatIndexMapping(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):

        if self.reverse == True:
            new_idxs = [
                i - self.offset[0]
                for i in idxs
                if (i >= self.offset[0] and i < self.offset[1])
            ]
        else:
            new_idxs = [i + self.offset[0] for i in idxs]
        return new_idxs


class _SplitIndexMapping(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse == True:
            new_idxs = [i + self.offset[0] for i in idxs]
        else:
            new_idxs = [
                i - self.offset[0]
                for i in idxs
                if (i >= self.offset[0] and i < self.offset[1])
            ]
        return new_idxs


class _GroupConvIndexMapping(object):
    def __init__(self, in_channels, out_channels, groups, reverse=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse == True:
            new_idxs = [i + self.offset[0] for i in idxs]
        else:
            group_histgram = np.histogram(
                idxs, bins=self.groups, range=(0, self.out_channels)
            )
            max_group_size = int(group_histgram.max())
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
