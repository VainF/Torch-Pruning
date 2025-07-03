"""Index mapping utility functions for managing pruning indices between layers."""
import torch
from .. import _helpers, ops
from .node import Node
from . import constants
from . import shape_infer

def update_index_mapping(graph):
    """ Update all index mapping in a graph
    """       
    # update index mapping
    for module, node in graph.module2node.items():
        if node.type == ops.OPTYPE.CONCAT:
            update_concat_index_mapping(graph, node)
        if node.type == ops.OPTYPE.SPLIT:
            update_split_index_mapping(graph, node)
        if node.type == ops.OPTYPE.RESHAPE:
            update_reshape_index_mapping(graph, node)
        if node.type == ops.OPTYPE.UNBIND:
            update_unbind_index_mapping(graph, node)
        if node.type == ops.OPTYPE.EXPAND and torch.__version__ >= "1.8":
            update_expand_index_mapping(graph, node)
        if node.type == ops.OPTYPE.SLICE:
            update_slice_index_mapping(graph, node)


class _FlattenIndexMapping(object):
    def __init__(self, stride=1, reverse=False):
        self._stride = stride
        self.reverse = reverse

    def __call__(self, idxs: _helpers._HybridIndex):
        new_idxs = []
        
        if self.reverse == True:
            for i in idxs:
                new_idxs.append( _helpers._HybridIndex( idx = (i.idx // self._stride), root_idx=i.root_idx ) )
            new_idxs = list(set(new_idxs))
        else:
            for i in idxs:
                new_idxs.extend(
                    [ _helpers._HybridIndex(idx=k, root_idx=i.root_idx) for k in range(i.idx * self._stride, (i.idx + 1) * self._stride) ]  
                )
        return new_idxs


class _ConcatIndexMapping(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs: _helpers._HybridIndex):
        if self.reverse == True:
            new_idxs = [
                _helpers._HybridIndex(idx = i.idx - self.offset[0], root_idx=i.root_idx )
                for i in idxs
                if (i.idx >= self.offset[0] and i.idx < self.offset[1])
            ]
        else:
            new_idxs = [ _helpers._HybridIndex(idx=i.idx + self.offset[0], root_idx=i.root_idx) for i in idxs]
        return new_idxs

class _GQAIndexMapping(object):
    def __init__(self, repeat, head_dim, reverse=False):
        self.repeat = repeat
        self.reverse = reverse
        self.head_dim = head_dim

    def __call__(self, idxs: _helpers._HybridIndex):
        head_dim = self.head_dim
        repeat = self.repeat
        if self.reverse == True: 
            new_idxs = [ _helpers._HybridIndex(idx=( i.idx - i.idx // (head_dim * repeat) * head_dim * (repeat - 1) - i.idx//head_dim%repeat * head_dim ), root_idx=None) for i in idxs ]
        else:
            new_idxs = []
           
        return new_idxs

class _SliceIndexMapping(object):
    def __init__(self, dim, start, step, end, reverse=False):
        self.start = start
        self.step = step
        self.end = end
        self.reverse = reverse
        self.dim = dim
    
    def __call__(self, idxs: _helpers._HybridIndex):
        
        if self.reverse == True:
            new_idxs = [ _helpers._HybridIndex(idx=i.idx * self.step + self.start, root_idx=i.root_idx) for i in idxs]
        else:
            new_idxs = [ _helpers._HybridIndex(idx=(i.idx - self.start) // self.step, root_idx=i.root_idx) for i in idxs if (i.idx >= self.start and i.idx < self.end and (i.idx-self.start)%self.step==0) ]
        return new_idxs

class _SplitIndexMapping(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs: _helpers._HybridIndex):
        if self.reverse == True:
            new_idxs = [ _helpers._HybridIndex(idx=i.idx + self.offset[0], root_idx=i.root_idx) for i in idxs]
        else:
            new_idxs = [
                _helpers._HybridIndex(idx = i.idx - self.offset[0], root_idx=i.root_idx)
                for i in idxs
                if (i.idx >= self.offset[0] and i.idx < self.offset[1])
            ]
        return new_idxs


def update_slice_index_mapping(graph, slice_node: Node):
    """Update index mapping for slice operations."""
    if slice_node.type != ops.OPTYPE.SLICE:
        return
    grad_fn = slice_node.grad_fn
    if hasattr(grad_fn, '_saved_self_sym_sizes'):
        if len(grad_fn._saved_self_sym_sizes)==4 and grad_fn._saved_dim != 1 and grad_fn._saved_dim<constants.MAX_VALID_DIM:
            return
        elif len(grad_fn._saved_self_sym_sizes)==3 and grad_fn._saved_dim != 2 and grad_fn._saved_dim<constants.MAX_VALID_DIM:
            return

    start, step, end, dim = slice_node.module.start, slice_node.module.step, slice_node.module.end, slice_node.module.dim
    for node in slice_node.inputs:
        for dep in slice_node.dependencies:
            if dep.target == node:
                dep.index_mapping[0] = _SliceIndexMapping(
                    dim=dim, start=start, end=end, step=step, reverse=True
                )
        for dep in node.dependencies:
            if dep.target == slice_node:
                dep.index_mapping[0] = _SliceIndexMapping(
                    dim=dim, start=start, end=end, step=step, reverse=False
                )

def update_flatten_index_mapping(graph, fc_node: Node):
    """Update index mapping for flatten operations."""
    if fc_node.type != ops.OPTYPE.LINEAR:
        return
    fc_in_features = fc_node.module.in_features
    feature_channels = 0
    for n in fc_node.inputs:
        recursive_depth = [0]
        feature_channels = shape_infer._infer_out_channels_recursively(graph, n, recursive_depth)
        if feature_channels is not None:
            break
                
    if feature_channels is None:  # the first layer: https://github.com/VainF/Torch-Pruning/issues/21
        return
    stride = fc_in_features // feature_channels
    if stride > 1 and fc_in_features % feature_channels == 0:
        for in_node in fc_node.inputs:
            for dep in fc_node.dependencies:
                if dep.target == in_node:
                    dep.index_mapping[0] = _FlattenIndexMapping(
                        stride=stride, reverse=True
                    )
            for dep in in_node.dependencies:
                if dep.target == fc_node:
                    dep.index_mapping[0] = _FlattenIndexMapping(
                        stride=stride, reverse=False
                    )

def update_reshape_index_mapping(graph, reshape_node: Node):
    
    # Only Supports 2D/4D tensors
    # TODO: Better support for reshape/view/flatten
    if hasattr(reshape_node.grad_fn, '_saved_self_sizes'): 
        size = reshape_node.grad_fn._saved_self_sizes
        if (len(size)!=1 and len(size)!=4):
            return
    else: # legacy version
        if not graph._2d_4d:
            return 

    out_channels = None
    for n in reshape_node.outputs:
        recursive_depth = [0]
        out_channels = shape_infer.infer_in_channels_recursively(graph, n, recursive_depth)
        if recursive_depth[0] > constants.MAX_RECURSION_DEPTH:
            return
        if out_channels is not None:  # =0 if there is a residual connection to model inputs
            break
    
    in_channels = None
    for n in reshape_node.inputs:
        recursive_depth = [0]
        in_channels = shape_infer.infer_out_channels_recursively(graph, n, recursive_depth)
        if recursive_depth[0] > constants.MAX_RECURSION_DEPTH:
            return
        if in_channels is not None:  # =0 if there is a residual connection to model inputs
            break
    
    if out_channels is None or in_channels is None: return
    if out_channels==in_channels: return
    
    if hasattr(reshape_node.grad_fn, '_saved_self_sizes'):
        if len(size)==4 and size[1]*size[2]*size[3]!=out_channels:
            return
    
    # Flatten
    if out_channels > in_channels:
         for in_node in reshape_node.inputs:
            for dep in reshape_node.dependencies:
                if dep.target == in_node:
                    dep.index_mapping[0] = _FlattenIndexMapping(
                        stride=out_channels // in_channels, reverse=True
                    )
            for dep in in_node.dependencies:
                if dep.target == reshape_node:
                    dep.index_mapping[0] = _FlattenIndexMapping(
                        stride=out_channels // in_channels, reverse=False
                    )
    else: # 1D -> 2D
        for out_node in reshape_node.outputs:
            for dep in reshape_node.dependencies:
                if dep.target == out_node:
                    dep.index_mapping[0] = _FlattenIndexMapping(
                        stride=in_channels // out_channels, reverse=True
                    )
            for dep in out_node.dependencies:
                if dep.target == reshape_node:
                    dep.index_mapping[0] = _FlattenIndexMapping(
                        stride=in_channels // out_channels, reverse=False
                    )
        
def update_concat_index_mapping(graph, cat_node: Node):
    if cat_node.type != ops.OPTYPE.CONCAT:
        return
        
    if hasattr(cat_node.grad_fn, '_saved_dim') and cat_node.grad_fn._saved_dim != 1 and cat_node.grad_fn._saved_dim < constants.MAX_VALID_DIM:
        return 

    if cat_node.module.concat_sizes is not None:
        chs = cat_node.module.concat_sizes
    else:
        chs = []
        for n in cat_node.inputs:
            chs.append(shape_infer.infer_channels_between(graph, n, cat_node))
        cat_node.module.concat_sizes = chs

    out_size = shape_infer.infer_in_channels_recursively(graph, cat_node, recursive_depth=[0])
    if out_size is not None:
        if out_size != sum(chs):
            return # the concat was applied on a different dimension than the feature dimension

    offsets = [0]
    for ch in chs:
        if ch is None: 
            return
        offsets.append(offsets[-1] + ch)
    cat_node.module.offsets = offsets

    # no transform if the concat dim is different from the feature dim
    # TODO: make the index mapping more flexible
    addressed_dep = []
    for i, in_node in enumerate(cat_node.inputs):
        for dep in cat_node.dependencies:
            if any((dep is d) for d in addressed_dep): continue
            if dep.target == in_node:
                if cat_node.enable_index_mapping:
                    dep.index_mapping[1] = _ConcatIndexMapping(
                        offset=offsets[i: i + 2], reverse=True
                    )
                    addressed_dep.append(dep)
                    break
                    
    addressed_dep = []
    for i, in_node in enumerate(cat_node.inputs):
        for dep in in_node.dependencies:
            if any((dep is d) for d in addressed_dep): continue
            if dep.target == cat_node:
                if cat_node.enable_index_mapping:
                    dep.index_mapping[1] = _ConcatIndexMapping(
                        offset=offsets[i: i + 2], reverse=False
                    )
                    addressed_dep.append(dep)
                    break

def update_split_index_mapping(graph, split_node: Node):
    if split_node.type != ops.OPTYPE.SPLIT:
        return

    if hasattr(split_node.grad_fn, '_saved_dim'): # this only works for Pytorch>=1.12
        # There a issue in some pytorch version, where the _saved_dim is an uninitialized value like 118745347895359
        # So we need to check if the _saved_dim is a valid value (<len(_saved_self_sym_sizes) or a nominal value like 20)
        if hasattr(split_node.grad_fn, '_saved_self_sym_sizes'):
            if split_node.grad_fn._saved_dim<len(split_node.grad_fn._saved_self_sym_sizes) and split_node.grad_fn._saved_dim != 1:
                return
        else:
            if split_node.grad_fn._saved_dim>=0 and split_node.grad_fn._saved_dim != 1:
                return 

    offsets = split_node.module.offsets

    if offsets is None:
        return
    addressed_dep = []
    for i, out_node in enumerate(split_node.outputs):
        for dep in split_node.dependencies:
            if any((dep is d) for d in addressed_dep): continue
            if dep.target == out_node:
                if split_node.enable_index_mapping:
                    dep.index_mapping[0] = (_SplitIndexMapping(
                        offset=offsets[i: i + 2], reverse=False
                    ))
                    addressed_dep.append(dep)
                    break
    
    addressed_dep = []
    for i, out_node in enumerate(split_node.outputs):
        for dep in out_node.dependencies:
            if dep.target == split_node:
                if any((dep is d) for d in addressed_dep): continue
                if split_node.enable_index_mapping:
                    dep.index_mapping[0] = (_SplitIndexMapping(
                        offset=offsets[i: i + 2], reverse=True
                    ))
                    addressed_dep.append(dep)
                    break

def update_unbind_index_mapping(graph, unbind_node: Node):
    if unbind_node.type != ops.OPTYPE.UNBIND:
        return

    if hasattr(unbind_node.grad_fn, '_saved_dim') and (unbind_node.grad_fn._saved_dim )!= 0: # this only works for Pytorch>=1.12
        return 

    num_chunks = len(unbind_node.outputs)

    for input_node in unbind_node.inputs:
        input_dims = shape_infer.infer_out_channels_recursively(graph, input_node, [0])
        if input_dims is not None:
            break
    if input_dims is None: return
    unbind_node.module.offsets = [i*input_dims//num_chunks for i in range(num_chunks+1)]

    offsets = unbind_node.module.offsets
    if offsets is None:
        return
    addressed_dep = []
    for i, out_node in enumerate(unbind_node.outputs):
        for dep in unbind_node.dependencies:
            if any((dep is d) for d in addressed_dep): continue
            if dep.target == out_node:
                if unbind_node.enable_index_mapping:
                    dep.index_mapping[0] = (_SplitIndexMapping(
                        offset=offsets[i: i + 2], reverse=False
                    ))
                    addressed_dep.append(dep)
                    break
    
    addressed_dep = []
    for i, out_node in enumerate(unbind_node.outputs):
        for dep in out_node.dependencies:
            if dep.target == unbind_node:
                if any((dep is d) for d in addressed_dep): continue
                if unbind_node.enable_index_mapping:
                    dep.index_mapping[0] = (_SplitIndexMapping(
                        offset=offsets[i: i + 2], reverse=True
                    ))
                    addressed_dep.append(dep)
                    break

def update_expand_index_mapping(graph, node: Node):
    out_channels = None
    for n in node.outputs:
        recursive_depth = [0]
        out_channels = shape_infer.infer_in_channels_recursively(graph, n, recursive_depth)
        if recursive_depth[0] > constants.MAX_RECURSION_DEPTH:
            return
        if out_channels is not None:  # =0 if there is a residual connection to model inputs
            break
    if not hasattr(node.grad_fn, '_saved_self_sym_sizes'):
        #warnings.warn("Expand operation detected but the shape information is not available")
        return 

    # for Huggingface GQA only, will support more expand operations in the future
    if len(node.grad_fn._saved_self_sym_sizes) == 5:
        batch, num_key_value_heads, n_rep, slen, head_dim = node.grad_fn._saved_self_sym_sizes
        in_channels = num_key_value_heads * n_rep * head_dim
        if out_channels is None or in_channels is None: return
        repeat = out_channels // in_channels
        addressed_dep = []

        for i, in_node in enumerate(node.inputs):
            for dep in node.dependencies:
                if any((dep is d) for d in addressed_dep): continue
                if dep.target == in_node:
                    if node.enable_index_mapping:
                        dep.index_mapping[0] = (_GQAIndexMapping(repeat=repeat, reverse=True, head_dim=head_dim))
                        addressed_dep.append(dep)
                        break
        
        addressed_dep = []
        for i, in_node in enumerate(node.inputs):
            for dep in in_node.dependencies:
                if dep.target == node:
                    if any((dep is d) for d in addressed_dep): continue
                    if node.enable_index_mapping:
                        dep.index_mapping[0] = (_GQAIndexMapping(repeat=repeat, reverse=False, head_dim=head_dim))
                        addressed_dep.append(dep)
                        break