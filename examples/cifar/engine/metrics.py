import numpy as np
import torch
from typing import Callable

__all__=['Accuracy', 'TopkAccuracy']

from abc import ABC, abstractmethod
from typing import Callable, Union, Any, Mapping, Sequence
import numbers
import numpy as np

class Metric(ABC):
    @abstractmethod
    def update(self, pred, target):
        """ Overridden by subclasses """
        raise NotImplementedError()
    
    @abstractmethod
    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

class MetricCompose(dict):
    def __init__(self, metric_dict: Mapping):
        self._metric_dict = metric_dict

    @property
    def metrics(self):
        return self._metric_dict
        
    @torch.no_grad()
    def update(self, outputs, targets):
        for key, metric in self._metric_dict.items():
            if isinstance(metric, Metric):
                metric.update(outputs, targets)
    
    def get_results(self):
        results = {}
        for key, metric in self._metric_dict.items():
            if isinstance(metric, Metric):
                results[key] = metric.get_results()
        return results

    def reset(self):
        for key, metric in self._metric_dict.items():
            if isinstance(metric, Metric):
                metric.reset()

    def __getitem__(self, name):
        return self._metric_dict[name]

class Accuracy(Metric):
    def __init__(self):
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs = outputs.max(1)[1]
        self._correct += ( outputs.view(-1)==targets.view(-1) ).sum()
        self._cnt += torch.numel( targets )

    def get_results(self):
        return (self._correct / self._cnt * 100.).detach().cpu()
    
    def reset(self):
        self._correct = self._cnt = 0.0


class TopkAccuracy(Metric):
    def __init__(self, topk=(1, 5)):
        self._topk = topk
        self.reset()
    
    @torch.no_grad()
    def update(self, outputs, targets):
        for k in self._topk:
            _, topk_outputs = outputs.topk(k, dim=1, largest=True, sorted=True)
            correct = topk_outputs.eq( targets.view(-1, 1).expand_as(topk_outputs) )
            self._correct[k] += correct[:, :k].view(-1).float().sum(0).item()
        self._cnt += len(targets)

    def get_results(self):
        return tuple( self._correct[k] / self._cnt * 100. for k in self._topk )

    def reset(self):
        self._correct = {k: 0 for k in self._topk}
        self._cnt = 0.0

class RunningLoss(Metric):
    def __init__(self, loss_fn, is_batch_average=False):
        self.reset()
        self.loss_fn = loss_fn
        self.is_batch_average = is_batch_average

    @torch.no_grad()
    def update(self, outputs, targets):
        self._accum_loss += self.loss_fn(outputs, targets)
        if self.is_batch_average:
            self._cnt += 1
        else:
            self._cnt += len(outputs)

    def get_results(self):
        return (self._accum_loss / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_loss = self._cnt = 0.0