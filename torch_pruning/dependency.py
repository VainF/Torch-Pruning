import torch
import torch.nn as nn
import typing
from functools import reduce
from operator import mul
from . import prune
from enum import IntEnum
from numbers import Number

__all__ = ['PruningPlan', 'Dependency', 'DependencyGraph']

TORCH_CONV = nn.modules.conv._ConvNd
TORCH_BATCHNORM = nn.modules.batchnorm._BatchNorm
TORCH_LAYERNORM = nn.modules.normalization.LayerNorm
TORCH_PRELU = nn.PReLU
TORCH_LINEAR = nn.Linear
TORCH_EMBED = nn.Embedding

class OPTYPE(IntEnum):
    CONV = 0
    BN = 1
    LINEAR = 2
    PRELU = 3
    GROUP_CONV=4

    CONCAT=5
    SPLIT=6
    CUSTOMIZED=7
    ELEMENTWISE=8

    LN = 9
    EMBED = 10

def _get_module_type(module):
    if isinstance( module, TORCH_CONV ):
        if module.groups>1:
            return OPTYPE.GROUP_CONV
        else:
            return OPTYPE.CONV
    elif isinstance( module, TORCH_BATCHNORM ):
        return OPTYPE.BN
    elif isinstance( module, TORCH_PRELU ):
        return OPTYPE.PRELU
    elif isinstance( module, TORCH_LINEAR ):
        return OPTYPE.LINEAR
    elif isinstance( module, _ConcatOp ):
        return OPTYPE.CONCAT
    elif isinstance( module, _SplitOp):
        return OPTYPE.SPLIT
    elif isinstance( module, TORCH_LAYERNORM ):
        return OPTYPE.LN
    elif isinstance( module, TORCH_EMBED ):
        return OPTYPE.EMBED
    elif isinstance(module, _CustomizedOp):
        return OPTYPE.CUSTOMIZED
    else:
        return OPTYPE.ELEMENTWISE

def _get_node_out_channel(node):
    if node.type==OPTYPE.CONV or node.type==OPTYPE.GROUP_CONV:
        return node.module.out_channels
    elif node.type==OPTYPE.BN:
        return node.module.num_features
    elif node.type==OPTYPE.LN:
        return node.module.normalized_shape[node.pruning_dim]
    elif node.type==OPTYPE.LINEAR:
        return node.module.out_features
    elif node.type==OPTYPE.PRELU:
        if node.module.num_parameters==1:
            return None
        else:
            return node.module.num_parameters
    elif node.type==OPTYPE.CUSTOMIZED:
        return node.customized_op_fn['get_out_ch_fn'](node.module)
    else:
        return None

def _get_node_in_channel(node):
    if node.type==OPTYPE.CONV or node.type==OPTYPE.GROUP_CONV:
        return node.module.in_channels
    elif node.type==OPTYPE.BN:
        return node.module.num_features
    elif node.type==OPTYPE.LN:
        return node.module.normalized_shape[node.pruning_dim]
    elif node.type==OPTYPE.LINEAR:
        return node.module.in_features
    elif node.type==OPTYPE.PRELU:
        if node.module.num_parameters==1:
            return None
        else:
            return node.module.num_parameters
    elif node.type==OPTYPE.CUSTOMIZED:
        return node.customized_op_fn['get_in_ch_fn'](node.module)
    else:
        return None

# Dummy Pruning fn
def _prune_concat(layer, *args, **kargs):
    return layer, 0

def _prune_split(layer, *args, **kargs):
    return layer, 0

def _prune_elementwise_op(layer, *args, **kargs):
    return layer, 0

class _CustomizedOp(nn.Module):
    def __init__(self, op_class):
        self.op_cls = op_class

    def __repr__(self):
        return "CustomizedOp(%s)"%(str(self.op_cls))

# Dummy module
class _ConcatOp(nn.Module):
    def __init__(self):
        super(_ConcatOp, self).__init__()
        self.offsets = None
        
    def __repr__(self):
        return "_ConcatOp(%s)"%(self.offsets)

class _SplitOp(nn.Module):
    def __init__(self):
        super(_SplitOp, self).__init__()
        self.offsets = None
        
    def __repr__(self):
        return "_SplitOp(%s)"%(self.offsets)

class _ElementWiseOp(nn.Module):
    def __init__(self):
        super(_ElementWiseOp, self).__init__()

    def __repr__(self):
        return "_ElementWiseOp()"

class _FlattenIndexTransform(object):
    def __init__(self, stride=1, reverse=False):
        self._stride = stride
        self.reverse = reverse

    def __call__(self, idxs):
        new_idxs = []
        if self.reverse==True:
            for i in idxs:
                new_idxs.append( i//self._stride )
                new_idxs = list( set(new_idxs) )
        else:
            for i in idxs:
                new_idxs.extend(list( range(i*self._stride, (i+1)*self._stride)))
        return new_idxs


class _ConcatIndexTransform(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse==True:
            new_idxs = [i-self.offset[0] for i in idxs if (i>=self.offset[0] and i<self.offset[1])]
        else:
            new_idxs = [i+self.offset[0] for i in idxs]
        return new_idxs


class _SplitIndexTransform(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse==True:
            new_idxs = [i+self.offset[0] for i in idxs ]
        else:
            new_idxs = [i-self.offset[0] for i in idxs if (i>=self.offset[0] and i<self.offset[1])]
        return new_idxs
        
class Node(object):
    def __init__(self, module, grad_fn, pruning_dim=-1, node_name=None):
        self.module = module
        self.grad_fn = grad_fn
        self.inputs = []
        self.outputs = []
        self.dependencies = []
        self.pruning_dim = pruning_dim
        self._node_name = node_name 
        self.type = _get_module_type( module )

    @property
    def node_name(self):
        return "%s (%s)"%(self._node_name, str(self.module)) if self._node_name is not None else str(self.module)

    def add_input(self, node):
        if node not in self.inputs:
            self.inputs.append( node )
    
    def add_output(self, node):
        if node not in self.outputs:
            self.outputs.append( node )

    def __repr__(self):
        return "<Node: (%s, %s)>"%( self.node_name, self.grad_fn )

    def __str__(self):
        return "<Node: (%s, %s)>"%( self.node_name, self.grad_fn )

    def details(self):
        fmt = "<Node: (%s, %s)>\n"%( self.node_name, self.grad_fn )
        fmt += ' '*4+'IN:\n'
        for in_node in self.inputs:
            fmt+=' '*8+'%s\n'%(in_node)
        fmt += ' '*4+'OUT:\n'
        for out_node in self.outputs:
            fmt+=' '*8+'%s\n'%(out_node)

        fmt += ' '*4+'DEP:\n'
        for dep in self.dependencies:
            fmt+=' '*8+"%s\n"%(dep)
        return fmt

class Dependency(object):
    def __init__(self, trigger, handler, broken_node: Node, index_transform: typing.Callable=None):
        """ Layer dependency in structed neural network pruning. 

        Parameters:
            trigger (Callable or None): a pruning function which will break the dependency 
            handler (Callable): a pruning function to fix the broken dependency
            broken_node (nn.Module): the broken layer
        """
        self.trigger = trigger
        self.handler = handler
        self.broken_node = broken_node
        self.index_transform = index_transform

    def __call__(self, idxs: list, dry_run: bool=False):
        result = self.handler(self.broken_node.module, idxs, pruning_dim=self.broken_node.pruning_dim, dry_run=dry_run)
        return result

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<DEP: %s => %s on %s>" % ("None" if self.trigger is None else self.trigger.__name__, self.handler.__name__, self.broken_node.node_name)

    def is_triggered_by(self, pruning_fn):
        return pruning_fn == self.trigger

    def __eq__(self, other):
        return ((self.trigger == other.trigger) and
                    self.handler == other.handler and 
                        self.broken_node == other.broken_node)

class PruningPlan(object):
    """ Pruning plan.
    
    Args:
        dry_run (Callable or None): only return the info about pruning.
        module_to_name (dict): mapping nn.module to a readable name. It will be filled by DependencyGraph.
    """

    def __init__(self):
        self._plans = list()

    def add_plan(self, dep, idxs):
        self._plans.append( (dep, idxs) )
    
    @property
    def plan(self):
        return self._plans

    def exec(self, dry_run=False):
        num_pruned = 0
        for dep, idxs in self._plans:
            _, n = dep(idxs, dry_run=dry_run)
            num_pruned += n
        return num_pruned

    def has_dep(self, dep):
        for _dep, _ in self._plans:
            if dep==_dep:
                return True    
        return False
    
    def has_pruning_op(self, dep, idxs):
        for _dep, _idxs in self._plans:
            if _dep.broken_node==dep.broken_node and _dep.handler==dep.handler and _idxs==idxs:
                return True
        return False

    def add_plan_and_merge(self, dep, idxs):
        for i, (_dep, _idxs) in enumerate(self._plans):
            if _dep.broken_node==dep.broken_node and _dep.handler==dep.handler:
                self._plans[i] = (_dep, list(set(_idxs+idxs)))
                return
        self.add_plan( dep, idxs )

    def __str__(self):
        fmt = ""
        fmt += "\n-------------\n"
        totally_pruned = 0
        for dep, idxs in self._plans:
            _, n_pruned = dep(idxs, dry_run=True)
            totally_pruned += n_pruned
            fmt += "[ %s, Index=%s, NumPruned=%d]\n" % (dep, idxs, n_pruned)
        fmt += "%d parameters will be pruned\n" % (totally_pruned)
        fmt += "-------------\n"
        return fmt


class DependencyGraph(object):

    PRUNABLE_MODULES = [ 
        nn.modules.conv._ConvNd, nn.modules.batchnorm._BatchNorm, nn.Linear, nn.PReLU,
        nn.modules.normalization.LayerNorm, nn.Embedding
    ]
    
    HANDLER = {    # pruning function that changes 1. in_channel           2. out_channel
                OPTYPE.CONV                 :  (prune.prune_related_conv,   prune.prune_conv),
                OPTYPE.BN                   :  (prune.prune_batchnorm,      prune.prune_batchnorm),
                OPTYPE.PRELU                :  (prune.prune_prelu,          prune.prune_prelu),
                OPTYPE.LINEAR               :  (prune.prune_related_linear, prune.prune_linear),
                OPTYPE.GROUP_CONV           :  (prune.prune_group_conv,     prune.prune_group_conv),
                OPTYPE.CONCAT               :  (_prune_concat,              _prune_concat),
                OPTYPE.SPLIT                :  (_prune_split,               _prune_split),
                OPTYPE.ELEMENTWISE          :  (_prune_elementwise_op,      _prune_elementwise_op),
                OPTYPE.LN                   :  (prune.prune_layernorm,      prune.prune_layernorm),
                OPTYPE.EMBED                :  (prune.prune_embedding,      prune.prune_embedding),
                OPTYPE.CUSTOMIZED           :  (None, None), # placeholder
            }
    OUTPUT_NODE_RULES = {}
    INPUT_NODE_RULES = {}
    for t1 in HANDLER.keys():
        for t2 in HANDLER.keys():
            OUTPUT_NODE_RULES[ (t1, t2) ] = (HANDLER[t1][1], HANDLER[t2][0]) # change in_channels of output layer
            INPUT_NODE_RULES[ (t1, t2) ] = (HANDLER[t1][0], HANDLER[t2][1])  # change out_channels of input layer
    CUSTOMIZED_OP_FN = {}

    def build_dependency( self, 
        model:torch.nn.Module, 
        example_inputs: typing.Union[torch.Tensor, typing.Sequence],
        pruning_dim: int=1,
        output_transform:typing.Callable=None, 
        verbose:bool=True ):
        """ Build a dependency graph through forwarding.

        Parameters:
            model (class): the model to be pruned.
            example_inputs (torch.Tensor or List): dummy inputs for the model.
            output_transform (Callable): A function to transform network outputs.
            verbose (Callable): verbose mode.
        """

        self.verbose = verbose
        # get module name
        self._module_to_name = { module: name for (name, module) in model.named_modules() }
        if pruning_dim >= 0:
            pruning_dim = pruning_dim - len(example_inputs.size())
        # build dependency graph
        self.module_to_node = self._obtain_forward_graph( model, example_inputs, output_transform=output_transform, pruning_dim=pruning_dim)
        self._build_dependency(self.module_to_node)
        self.update_index()
        return self

    def register_customized_layer(self, layer_type, in_ch_pruning_fn, out_ch_pruning_fn, get_in_ch_fn, get_out_ch_fn):
        """ Register a customized layer for pruning.

        Parameters:
            layer_type (class): the type of layer
            in_ch_pruning_fn (Callable): A function to prune channels/dimensions of input tensor
            out_ch_pruning_fn (Callable): A function to prune channels/dimensions of output tensor
            get_in_ch_fn (Callable): estimate the n_channel of layer input. Return None if the layer does not change tensor shape.
            get_out_ch_fn (Callable):estimate the n_channel of layer output. Return None if the layer does not change tensor shape.
        """
        self.CUSTOMIZED_OP_FN[layer_type] = {
            "in_ch_pruning_fn": in_ch_pruning_fn, 
            "out_ch_pruning_fn": out_ch_pruning_fn, 
            "get_in_ch_fn": get_in_ch_fn, 
            "get_out_ch_fn": get_out_ch_fn,
        }
        self.PRUNABLE_MODULES.append( layer_type )
 
    def get_pruning_plan(self, module: nn.Module, pruning_fn: typing.Callable, idxs: typing.Union[list, tuple]): 
        """ Get a pruning plan from the dependency graph, according to user's pruning operations. 

        Parameters:
            module (nn.Module): the module to be pruned.
            pruning_fn (Callable): the pruning function.
            idxs (list or tuple): the indices of paramters to be pruned.
        """
        if isinstance(module, TORCH_CONV) and module.groups>1:
            pruning_fn = prune.prune_group_conv
        if isinstance(idxs, Number):
            idxs = [idxs]

        self.update_index()
        plan = PruningPlan()
        #  the user pruning operation
        root_node = self.module_to_node[module]
        plan.add_plan(Dependency(pruning_fn, pruning_fn, root_node), idxs)
        
        visited = set()
        def _fix_denpendency_graph(node, fn, indices):
            visited.add( node )
            for dep in node.dependencies:
                if dep.is_triggered_by(fn): #and dep.broken_node not in visited:
                    if dep.index_transform is not None:
                        new_indices = dep.index_transform(indices)
                    else:
                        new_indices = indices

                    if len(new_indices)==0:
                        continue
                    if dep.broken_node in visited and plan.has_pruning_op( dep, new_indices ):
                        continue
                    else:
                        plan.add_plan(dep, new_indices)
                        _fix_denpendency_graph(dep.broken_node, dep.handler, new_indices)
        _fix_denpendency_graph(root_node, pruning_fn, idxs)

        # merge pruning ops
        merged_plan = PruningPlan()
        for dep, idxs in plan.plan:
            merged_plan.add_plan_and_merge( dep, idxs )
        return merged_plan

    def _build_dependency(self, module_to_node):
        for module, node in module_to_node.items():
            for in_node in node.inputs:
                in_node_rule = self.INPUT_NODE_RULES.get( (node.type, in_node.type), None )
                if in_node_rule is not None:
                    trigger = in_node_rule[0]
                    handler = in_node_rule[1]
                    if trigger is None:
                        trigger = self.CUSTOMIZED_OP_FN[type(node.module)]['in_ch_pruning_fn']
                    if handler is None:
                        handler = self.CUSTOMIZED_OP_FN[type(in_node.module)]['out_ch_pruning_fn']
                    dep = Dependency( trigger=trigger, handler=handler, broken_node=in_node)
                    node.dependencies.append( dep )

            for out_node in node.outputs:
                out_node_rule = self.OUTPUT_NODE_RULES.get( (node.type, out_node.type), None )
                if out_node_rule is not None:
                    trigger = out_node_rule[0]
                    handler = out_node_rule[1]
                    if trigger is None:
                        trigger = self.CUSTOMIZED_OP_FN[type(node.module)]['out_ch_pruning_fn']
                    if handler is None:
                        handler = self.CUSTOMIZED_OP_FN[type(out_node.module)]['in_ch_pruning_fn']
                    dep = Dependency( trigger=trigger, handler=handler, broken_node=out_node)
                    node.dependencies.append( dep )
    
    def _obtain_forward_graph(self, model, example_inputs, output_transform, pruning_dim):
        #module_to_node = { m: Node( m ) for m in model.modules() if isinstance( m, self.PRUNABLE_MODULES ) }
        model.eval().cpu()
        # Get grad_fn from prunable modules
        grad_fn_to_module = {}

        visited = {}
        def _record_module_grad_fn(module, inputs, outputs):
            if module not in visited:
                visited[module] = 1
            else:
                visited[module] += 1
            grad_fn_to_module[outputs.grad_fn] = module
        
        hooks = [m.register_forward_hook(_record_module_grad_fn) for m in model.modules() if isinstance( m, tuple(self.PRUNABLE_MODULES )) ]
        
        if isinstance(example_inputs, (tuple, list)):
            out = model(*example_inputs)
        elif isinstance(example_inputs, dict):
            out = model(**example_inputs)
        elif isinstance(example_inputs, torch.Tensor):
            out = model(example_inputs)

        for hook in hooks:
            hook.remove()
        reused = [ m for (m, count) in visited.items() if count>1 ]
        # create nodes and dummy modules
        module_to_node = {}
        def _build_graph(grad_fn):
            module = grad_fn_to_module.get( grad_fn, None )   
            if module is not None and module in module_to_node and module not in reused:
                return module_to_node[module]
            
            if module is None:
                if not hasattr(grad_fn, 'name'):
                    module = _ElementWiseOp() # skip customized modules
                    if self.verbose:
                        print("[Warning] Unrecognized operation: %s. It will be treated as element-wise op"%( str(grad_fn) ))
                elif 'catbackward' in grad_fn.name().lower(): # concat op
                    module = _ConcatOp()
                elif 'splitbackward' in grad_fn.name().lower():
                    module = _SplitOp()
                else:
                    module = _ElementWiseOp()   # All other ops are treated as element-wise ops
                grad_fn_to_module[ grad_fn ] = module # record grad_fn

            if module not in module_to_node:
                node = Node( module, grad_fn, pruning_dim, self._module_to_name.get( module, None ) )
                if type(module) in self.CUSTOMIZED_OP_FN.keys(): # mark it as a customized OP
                    node.type = OPTYPE.CUSTOMIZED
                    node.customized_op_fn = self.CUSTOMIZED_OP_FN[type(module)]
                module_to_node[ module ] = node
            else:
                node = module_to_node[module]

            if hasattr(grad_fn, 'next_functions'):
                for f in grad_fn.next_functions:
                    if f[0] is not None:
                        if hasattr( f[0], 'name' ) and 'accumulategrad' in f[0].name().lower(): # skip leaf variables
                            continue
                        input_node = _build_graph(f[0])
                        node.add_input( input_node )
                        input_node.add_output( node )
            return node
        
        if output_transform is not None:
            out = output_transform(out)
        
        #if isinstance(out, (list, tuple) ):
        #    for o in out:
        #        _build_graph( o.grad_fn )
        #else:
        #    _build_graph( out.grad_fn )
        for o in flatten_as_list(out):
            _build_graph( o.grad_fn )
        return module_to_node

    def update_index( self ):
        for module, node in self.module_to_node.items():
            if node.type==OPTYPE.LINEAR:
                self._set_fc_index_transform( node )
            if node.type==OPTYPE.CONCAT:
                self._set_concat_index_transform(node)
            if node.type==OPTYPE.SPLIT:
                self._set_split_index_transform(node)

    def _set_fc_index_transform(self, fc_node: Node):
        if fc_node.type != OPTYPE.LINEAR:
            return
        visited = set()
        fc_in_features = fc_node.module.in_features
        feature_channels = _get_out_channels_of_in_node(fc_node.inputs[0])
        if feature_channels<=0: # the first layer: https://github.com/VainF/Torch-Pruning/issues/21
            return
        stride = fc_in_features // feature_channels
        if stride>1:
            for in_node in fc_node.inputs:
                for dep in fc_node.dependencies:
                    if dep.broken_node==in_node:
                        dep.index_transform = _FlattenIndexTransform( stride=stride, reverse=True )
                
                for dep in in_node.dependencies:
                    if dep.broken_node == fc_node:
                        dep.index_transform = _FlattenIndexTransform( stride=stride, reverse=False )

    def _set_concat_index_transform(self, cat_node: Node):
        if cat_node.type != OPTYPE.CONCAT:
            return
        
        chs = []
        for n in cat_node.inputs:
            chs.append( _get_out_channels_of_in_node(n) )

        offsets = [0]
        for ch in chs:
            offsets.append( offsets[-1]+ch )
        cat_node.module.offsets = offsets

        for i, in_node in enumerate(cat_node.inputs):
            for dep in cat_node.dependencies:
                if dep.broken_node == in_node:
                    dep.index_transform = _ConcatIndexTransform( offset=offsets[i:i+2], reverse=True )

            for dep in in_node.dependencies:
                if dep.broken_node == cat_node:
                    dep.index_transform = _ConcatIndexTransform( offset=offsets[i:i+2], reverse=False )

    def _set_split_index_transform(self, split_node: Node):
        if split_node.type != OPTYPE.SPLIT:
            return
        
        chs = []
        for n in split_node.outputs:
            chs.append( _get_in_channels_of_out_node(n) )

        offsets = [0]
        for ch in chs:
            offsets.append( offsets[-1]+ch )
        split_node.module.offsets = offsets
        for i, out_node in enumerate(split_node.outputs):
            for dep in split_node.dependencies:
                if dep.broken_node == out_node:
                    dep.index_transform = _SplitIndexTransform( offset=offsets[i:i+2], reverse=False )

            for dep in out_node.dependencies:
                if dep.broken_node == split_node:
                    dep.index_transform = _SplitIndexTransform( offset=offsets[i:i+2], reverse=True )

def _get_out_channels_of_in_node(node):
    ch = _get_node_out_channel(node)
    if ch is None:
        ch = 0
        for in_node in node.inputs:
            if node.type==OPTYPE.CONCAT:
                ch+=_get_out_channels_of_in_node( in_node )
            else:
                ch=_get_out_channels_of_in_node( in_node )
    return ch

def _get_in_channels_of_out_node(node):
    ch = _get_node_in_channel(node)
    if ch is None:
        ch = 0
        for out_node in node.outputs:
            if node.type==OPTYPE.SPLIT:
                ch+=_get_in_channels_of_out_node( out_node )
            else:
                ch=_get_in_channels_of_out_node( out_node )
    return ch

def flatten_as_list(obj):
    if isinstance(obj, torch.Tensor):
        return [ obj ]
    elif isinstance(obj, (list,tuple)):
        flattened_list = []
        for sub_obj in obj:
            flattened_list.extend( flatten_as_list(sub_obj) )
        return flattened_list
    elif isinstance(obj, dict):
        flattened_list = []
        for sub_obj in obj.values():
            flattened_list.extend( flatten_as_list(sub_obj) )
        return flattened_list
    else:
        return obj
    