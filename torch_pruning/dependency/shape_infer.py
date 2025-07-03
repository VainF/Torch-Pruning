"""Shape inference utility functions."""
import warnings
from . import constants
from .node import Node
from .. import ops

def init_shape_information(graph):
    for module, node in graph.module2node.items():

        if node.type == ops.OPTYPE.SPLIT:
            grad_fn = node.grad_fn

            if hasattr(grad_fn, '_saved_self_sizes') or hasattr(grad_fn, '_saved_split_sizes'):
                if hasattr(grad_fn, '_saved_split_sizes') and hasattr(grad_fn, '_saved_dim') :
                    if grad_fn._saved_dim != 1 and grad_fn._saved_dim < constants.MAX_VALID_DIM: # a temp fix for pytorch==1.11, where the _saved_dim is an uninitialized value like 118745347895359
                        continue
                    chs = list(grad_fn._saved_split_sizes)
                    node.module.split_sizes = chs
                elif hasattr(grad_fn, '_saved_split_size') and hasattr(grad_fn, '_saved_dim'):
                    if grad_fn._saved_dim != 1 and grad_fn._saved_dim < constants.MAX_VALID_DIM: # a temp fix for pytorch==1.11, where the _saved_dim is an uninitialized value like 118745347895359
                        continue
                    chs = [grad_fn._saved_split_size for _ in range(len(node.outputs))]
                    node.module.split_sizes = chs
                offsets = [0]
                for i in range(len(chs)):
                    offsets.append(offsets[i] + chs[i])
                node.module.offsets = offsets
            else: # legency version
                chs = []
                for n in node.outputs:
                    recursive_depth = [0]
                    chs.append(infer_in_channels_recursively(graph, n, recursive_depth))
                offsets = [0]
                for ch in chs:
                    if ch is None: continue
                    offsets.append(offsets[-1] + ch)
                node.module.split_sizes = chs
                node.module.offsets = offsets
                
def infer_out_channels_recursively(graph, node: Node, recursive_depth: list):
    """ Infer the number of output channels recursively
    
    Args:
        graph: The dependency graph
        node: The node to infer output channels for
        recursive_depth: A list containing the current recursion depth
        
    Returns:
        int or None: Number of output channels or None if it cannot be determined
    """     
    if recursive_depth[0] > constants.MAX_RECURSION_DEPTH:
        warnings.warn("Maximum recursive depth reached!")
        return None
    ch = graph.get_out_channels(node)
    if ch is None:
        ch = 0
        for in_node in node.inputs:
            if node.type == ops.OPTYPE.CONCAT:
                recursive_depth[0]+=1
                sub_ch = infer_out_channels_recursively(graph, in_node, recursive_depth)
                if sub_ch is None:
                    return None
                ch += sub_ch
            else:
                if in_node.type == ops.OPTYPE.SPLIT and in_node.module.split_sizes is not None:
                    for i, split_out_node in enumerate(in_node.outputs):
                        if split_out_node == node:
                            ch = in_node.module.split_sizes[i]
                else:
                    recursive_depth[0]+=1
                    ch = infer_out_channels_recursively(graph, in_node, recursive_depth)
        if ch == 0:
            return None
    return ch

def infer_in_channels_recursively(graph, node: Node, recursive_depth: list):
    """ Infer the number of input channels recursively
    
    Args:
        graph: The dependency graph
        node: The node to infer input channels for
        recursive_depth: A list containing the current recursion depth
        
    Returns:
        int or None: Number of input channels or None if it cannot be determined
    """         
    if recursive_depth[0] > constants.MAX_RECURSION_DEPTH:
        return None
    ch = graph.get_in_channels(node)
    if ch is None:
        ch = 0
        for out_node in node.outputs:
            if node.type == ops.OPTYPE.SPLIT:
                recursive_depth[0]+=1
                sub_ch = infer_in_channels_recursively(graph, out_node, recursive_depth)
                if sub_ch is None:
                    return None
                ch += sub_ch
            if out_node.type == ops.OPTYPE.CONCAT:
                concat_output_channels = infer_in_channels_recursively(graph, out_node, recursive_depth)
                sibling_input_channels = []
                for in_node in out_node.inputs:
                    if in_node != node:
                        s = infer_out_channels_recursively(graph, in_node, recursive_depth)
                        if s is not None:
                            sibling_input_channels.append(s)
                if concat_output_channels is None or len(sibling_input_channels) == 0:
                    return None
                return concat_output_channels - sum(sibling_input_channels)
            else:
                recursive_depth[0]+=1
                ch = infer_in_channels_recursively(graph, out_node, recursive_depth)
        if ch == 0:
            return None
    return ch


def infer_channels_between(graph, node_1, node_2):
    if node_1.type == ops.OPTYPE.SPLIT:
        for i, n in enumerate(node_1.outputs):
            if n == node_2:
                return node_1.module.split_sizes[i]
    recursive_depth = [0]
    return infer_out_channels_recursively(graph, node_1, recursive_depth)