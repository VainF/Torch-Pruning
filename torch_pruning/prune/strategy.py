import torch
from abc import abstractclassmethod, ABC
from typing import Sequence
import random
import warnings

def round_pruning_amount(total_parameters, n_to_prune, round_to):
    """round the parameter amount after pruning to an integer multiple of `round_to`.
    """
    round_to = int(round_to)
    if round_to<=1: return n_to_prune
    after_pruning = total_parameters - n_to_prune
    compensation = after_pruning % round_to
    #   round to the nearest (round_to * N)                           # avoid negative results
    if (compensation < round_to // 2 and after_pruning > round_to) or round_to>n_to_prune: 
        n_to_prune = n_to_prune + compensation # floor
    else:
        n_to_prune = n_to_prune - round_to + compensation # ceiling
    return n_to_prune

class BaseStrategy(ABC):
    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    @abstractclassmethod
    def apply(self, weights, amount=0.0, round_to=1)->  Sequence[int]:  # return index
        """ Apply the strategy on weights with user specified pruning percentage.

        Parameters:
            weights (torch.Parameter): weights to be pruned.
            amount (Callable): the percentage of weights to be pruned (amount<1.0) or the amount of weights to be pruned (amount>=1.0) 
            round_to (int): the number to which the number of pruned channels is rounded.
        """
        raise NotImplementedError

class RandomStrategy(BaseStrategy):

    def apply(self, weights, amount=0.0, round_to=1)->  Sequence[int]:  # return index
        if amount<=0: return []
        n = len(weights)
        n_to_prune = int(amount*n) if amount<1.0 else amount
        n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
        if n_to_prune == 0: return []
        indices = random.sample( list( range(n) ), k=n_to_prune )
        return indices

class LNStrategy(BaseStrategy):
    def __init__(self, p):
        self.p = p

    def apply(self, weights, amount=0.0, round_to=1)->  Sequence[int]:  # return index
        if amount<=0: return []
        n = len(weights)
        l1_norm = torch.norm( weights.view(n, -1), p=self.p, dim=1 )
        n_to_prune = int(amount*n) if amount<1.0 else amount 
        n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
        if n_to_prune == 0: return []
        threshold = torch.kthvalue(l1_norm, k=n_to_prune).values 
        indices = torch.nonzero(l1_norm <= threshold).view(-1).tolist()
        return indices

class L1Strategy(LNStrategy):
    def __init__(self):
        super(L1Strategy, self).__init__(p=1)

class L2Strategy(LNStrategy):
    def __init__(self):
        super(L2Strategy, self).__init__(p=2)
