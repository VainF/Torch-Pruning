import torch
from abc import abstractclassmethod, ABC
from typing import Sequence
import random

class BaseStrategy(ABC):
    def __call__(self, weights, amount=0.0):
        return self.apply(weights, amount=amount)

    @abstractclassmethod
    def apply(self, weights, amount=0.0, round_to=1)->  Sequence[int]:  # return index
        """ Apply the strategy on weights with user specified pruning percentage.

        Parameters:
            weights (torch.Parameter): weights to be pruned.
            amount (Callable): the percentage of weights to be pruned.
            round_to (int): the number to which the number of pruned channels is rounded.
        """
        raise NotImplementedError

class RandomStrategy(BaseStrategy):

    def apply(self, weights, amount=0.0, round_to=1)->  Sequence[int]:  # return index
        if amount<=0: return []
        n = len(weights)
        if round_to == 1:
            n_to_prune = int(amount * n)
        else:
            if n <= round_to:
                n_to_prune = 0
                print("Warning, initial number of channels is less than `round_to` parameter.")
            else:
                remainder = n % round_to
                n_to_prune = int(remainder + round(amount * n / round_to) * round_to)
        indices = random.sample( list( range(n) ), k=n_to_prune )
        return indices

class LNStrategy(BaseStrategy):
    def __init__(self, p):
        self.p = p

    def apply(self, weights, amount=0.0, round_to=1)->  Sequence[int]:  # return index
        if amount<=0: return []
        n = len(weights)
        l1_norm = torch.norm( weights.view(n, -1), p=self.p, dim=1 )
        if round_to == 1:
            n_to_prune = int(amount*n)
        else:
            if n <= round_to:
                n_to_prune = 0
                print("Warning, initial number of channels is less than `round_to` parameter.")
            else:
                remainder = n % round_to
                n_to_prune = int(remainder + round(amount*n/round_to)*round_to)
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