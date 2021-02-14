import torch
from abc import abstractclassmethod, ABC
from typing import Sequence
import random

class BaseStrategy(ABC):
    def __call__(self, weights, amount=0.0):
        return self.apply(weights, amount=amount)

    @abstractclassmethod
    def apply(self, weights, amount=0.0)->  Sequence[int]:  # return index
        raise NotImplementedError

class RandomStrategy(BaseStrategy):

    def apply(self, weights, amount=0.0)->  Sequence[int]:  # return index
        if amount<=0: return []
        n = len(weights)
        n_to_prune = int(amount*n)
        indices = random.sample( list( range(n) ), k=n_to_prune )
        return indices

class LNStrategy(BaseStrategy):
    def __init__(self, p):
        self.p = p

    def apply(self, weights, amount=0.0)->  Sequence[int]:  # return index
        if amount<=0: return []
        n = len(weights)
        l1_norm = torch.norm( weights.view(n, -1), p=self.p, dim=1 )
        n_to_prune = int(amount*n)
        if n_to_prune == 0:
            return []
        threshold = torch.kthvalue(l1_norm, k=n_to_prune).values 
        indices = torch.nonzero(l1_norm <= threshold).view(-1).tolist()
        return indices

class L1Strategy(LNStrategy):
    def __init__(self):
        super(L1Strategy, self).__init__(p=1)

class L2Strategy(LNStrategy):
    def __init__(self):
        super(L2Strategy, self).__init__(p=2)