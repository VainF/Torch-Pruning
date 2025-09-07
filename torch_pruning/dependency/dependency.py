"""Dependency class implementation for modeling layer dependencies."""

import typing

from .. import _helpers
from . import constants
from .node import Node

class Dependency(object):
    """Layer dependency (Edge of DepGraph).

    For the dependency A -> B, the pruning operation ``trigger(A)`` will trigger 
    the pruning operation ``handler(B)``.

    The object is callable, which will invoke the handler function for pruning.

        Args:
            trigger (Callable): a pruning function that triggers this dependency
            handler (Callable): a pruning function that can fix the broken dependency
            source (Node): the source node pruned by the trigger function
            target (Node): the target node pruned by the handler function
            index_mapping (Callable): a callable function for index mapping
    """
    def __init__(
        self,
        trigger: typing.Callable,
        handler: typing.Callable,
        source: Node,
        target: Node,
    ):
        self.trigger = trigger
        self.handler = handler
        self.source = source
        self.target = target             
        # index_mapping are used to map the indices of the source node to the target node
        # There will be two index_mapping functions for each dependency, to handle cascaded concat & split operations.
        # E.g. split -> concat
        # We first map the indeces to the splited tensor with index_mapping[0], 
        # then map the splited tensor to the concatenated tensor with index_mapping[1].
        # Current coordinate system => Standard coordinate system => target coordinate system 
        #                     index_mapping[0]           index_mapping[1]
        self.index_mapping = [constants.INDEX_MAPPING_PLACEHOLDER, constants.INDEX_MAPPING_PLACEHOLDER] # [None, None] by default

    def __call__(self, idxs: list):
        """Execute the dependency by calling the handler function.
        
        Args:
            idxs: List of indices to prune.
            
        Returns:
            The result of the handler function.
        """
        self.handler.__self__.pruning_dim = self.target.pruning_dim  # set pruning_dim
        if len(idxs) > 0 and isinstance(idxs[0], _helpers._HybridIndex): 
            # hybrid indices include root indices. We need to remove them and only pass the plain indices to the handler
            idxs = _helpers.to_plain_idxs(idxs)
        result = self.handler(self.target.module, idxs)
        return result

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "{} on {} => {} on {}".format(
            "None" if self.trigger is None else self.trigger.__name__,
            self.source.name,
            self.handler.__name__,
            self.target.name,
        )

    def is_triggered_by(self, pruning_fn):
        """Check if the dependency is triggered by a specific pruning function.
        
        Args:
            pruning_fn: The pruning function to check.
            
        Returns:
            True if the dependency is triggered by the given function.
        """
        return pruning_fn == self.trigger

    def __eq__(self, other):
        """Check if two dependencies are the same.
        
        Args:
            other: Another dependency to compare with.
            
        Returns:
            True if the dependencies are equal.
        """
        return (
            self.source == other.source 
            and self.trigger == other.trigger
            and self.handler == other.handler
            and self.target == other.target
        )
    
    @property
    def layer(self):
        """Alias of the target module."""
        return self.target.module

    @property
    def pruning_fn(self):
        """Alias of the handler."""
        return self.handler

    def __hash__(self):
        return hash((self.source, self.target, self.trigger, self.handler))