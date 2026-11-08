"""Shape inference utility functions."""
import warnings
from . import constants
from .node import Node
from .. import ops, _helpers

def acts_on_feature_dim(saved_dim, ndim):
    """Whether a cat/split/chunk recorded with ``saved_dim`` on a ``ndim``-D tensor may act on the feature dim.

    The feature dim is dim 1, or the last dim of (N, C) / (N, L, C) tensors such as nn.Linear outputs.
    For example, ``torch.cat((-x2, x1), dim=-1)`` in rotary embeddings acts on the head dim of a
    (B, H, N, D) tensor and must not shift the channel indices (#532).
    Unknown values return True, as before: the caller then checks the channel numbers.
    """
    if ndim is None:  # the rank is unknown (old PyTorch): legacy check on the raw value
        return saved_dim == 1 or saved_dim >= constants.MAX_VALID_DIM
    dim = _helpers.normalize_saved_dim(saved_dim, ndim)
    if not 0 <= dim < ndim:  # an uninitialized value, e.g. 118745347895359 in PyTorch 1.11
        return True
    return dim == 1 or (2 <= ndim <= 3 and dim == ndim - 1)

def init_shape_information(graph):
    for module, node in graph.module2node.items():

        if node.type == ops.OPTYPE.SPLIT:
            grad_fn = node.grad_fn

            if hasattr(grad_fn, '_saved_self_sizes') or hasattr(grad_fn, '_saved_split_sizes'):
                sizes = getattr(grad_fn, '_saved_self_sym_sizes', getattr(grad_fn, '_saved_self_sizes', None))
                ndim = len(sizes) if sizes is not None else None
                if hasattr(grad_fn, '_saved_split_sizes') and hasattr(grad_fn, '_saved_dim') :
                    if not acts_on_feature_dim(grad_fn._saved_dim, ndim):
                        continue
                    chs = list(grad_fn._saved_split_sizes)
                    node.module.split_sizes = chs
                elif hasattr(grad_fn, '_saved_split_size') and hasattr(grad_fn, '_saved_dim'):
                    if not acts_on_feature_dim(grad_fn._saved_dim, ndim):
                        continue
                    num_chunks = _helpers.grad_fn_num_outputs(grad_fn) or len(set(node.output_nr(k) for k in range(len(node.outputs))))
                    chs = [grad_fn._saved_split_size for _ in range(num_chunks)]
                    node.module.split_sizes = chs
                offsets = [0]
                for i in range(len(chs)):
                    offsets.append(offsets[i] + chs[i])
                node.module.offsets = offsets
            else: # legency version
                # one entry per chunk: a chunk consumed by several ops appears several times in node.outputs
                chunk_chs = {}
                for k, n in enumerate(node.outputs):
                    nr = node.output_nr(k)
                    if chunk_chs.get(nr, None) is not None:
                        continue
                    recursive_depth = [0]
                    chunk_chs[nr] = infer_in_channels_recursively(graph, n, recursive_depth)
                chs = [chunk_chs[nr] for nr in sorted(chunk_chs)]
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
                elif in_node.type == ops.OPTYPE.UNBIND and _helpers.grad_fn_num_outputs(in_node.grad_fn):
                    # each output of torch.unbind, e.g. q, k and v of a stacked qkv, gets an equal share of the channels
                    recursive_depth[0]+=1
                    ch = infer_out_channels_recursively(graph, in_node, recursive_depth)
                    if ch is not None:
                        ch = ch // _helpers.grad_fn_num_outputs(in_node.grad_fn)
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
        counted_chunks = set()
        for k, out_node in enumerate(node.outputs):
            if node.type == ops.OPTYPE.SPLIT:
                if node.output_nr(k) in counted_chunks: # this chunk feeds several ops
                    continue
                counted_chunks.add(node.output_nr(k))
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