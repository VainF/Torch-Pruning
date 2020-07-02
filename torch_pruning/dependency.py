import torch
import torch.nn as nn
import typing
from functools import reduce
from operator import mul
from . import prune
from enum import IntEnum

__all__ = ['PruningPlan', 'Dependency', 'DependencyGraph']

TORCH_CONV = nn.modules.conv._ConvNd
TORCH_BATCHNORM = nn.modules.batchnorm._BatchNorm
TORCH_PRELU = nn.PReLU
TORCH_LINEAR = nn.Linear

class OPTYPE(IntEnum):
    CONV = 0
    BN = 1
    LINEAR = 2
    PRELU = 3

    CONCAT=4
    ELEMENTWISE=5


def _get_module_type(module):
    if isinstance( module, TORCH_CONV ):
        return OPTYPE.CONV
    elif isinstance( module, TORCH_BATCHNORM ):
        return OPTYPE.BN
    elif isinstance( module, TORCH_PRELU ):
        return OPTYPE.PRELU
    elif isinstance( module, TORCH_LINEAR ):
        return OPTYPE.LINEAR
    elif isinstance( module, _ConcatOp ):
        return OPTYPE.CONCAT
    else:
        return OPTYPE.ELEMENTWISE

def _get_node_channel(node):
    if node.type==OPTYPE.CONV:
        return node.module.out_channels
    elif node.type==OPTYPE.BN:
        return node.module.num_features
    elif node.type==OPTYPE.LINEAR:
        return node.module.out_features
    elif node.type==OPTYPE.PRELU:
        if node.module.num_parameters==1:
            return None
        else:
            return node.module.num_parameters
    else:
        return None

# Dummy Pruning fn
def _prune_concat(layer, *args, **kargs):
    return layer, 0

def _prune_elementwise_op(layer, *args, **kargs):
    return layer, 0

# Dummy module
class _ConcatOp(nn.Module):
    def __init__(self):
        super(_ConcatOp, self).__init__()
        self.offsets = None
        
    def __repr__(self):
        return "_ConcatOp(%s)"%(self.offsets)

class _ElementWiseOp(nn.Module):
    def __init__(self):
        super(_ElementWiseOp, self).__init__()

    def __repr__(self):
        return "_ElementWiseOp()"

class _Inputs(nn.Module):
    def __init__(self):
        super(_Inputs, self).__init__()

class _StrideIndexTransform(object):
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

class _OffsetIndexTransform(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse==True:
            new_idxs = [i-self.offset[0] for i in idxs if (i>=self.offset[0] and i<self.offset[1])]
        else:
            new_idxs = [i+self.offset[0] for i in idxs]
        return new_idxs
        
class Node(object):
    def __init__(self, module, grad_fn, node_name=None):
        self.module = module
        self.grad_fn = grad_fn
        self.inputs = []
        self.outputs = []
        self.dependencies = []
        self._node_name = node_name 
        self.type = _get_module_type( module )

    @property
    def node_name(self):
        return "%s (%s)"%(self._node_name, str(self.module)) if self._node_name is not None else str(self.module)

    def add_input(self, node):
        self.inputs.append( node )
    
    def add_output(self, node):
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
        result = self.handler(self.broken_node.module, idxs, dry_run=dry_run)
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

    PRUNABLE_MODULES = ( nn.modules.conv._ConvNd, nn.modules.batchnorm._BatchNorm, nn.Linear, nn.PReLU )
    ELEMENTWISE_GRAD_FN = ('AddBackward', 'SubBackward', 'MulBackward', 'DivBackward', 'PowBackward', )
    CONCAT_GRAD_FN = ('CatBackward',)
    
    HANDLER = {                         # prune in_channel          # prune out_channel
                OPTYPE.CONV          :  (prune.prune_related_conv,   prune.prune_conv),
                OPTYPE.BN            :  (prune.prune_batchnorm,      prune.prune_batchnorm),
                OPTYPE.PRELU         :  (prune.prune_prelu,          prune.prune_prelu),
                OPTYPE.LINEAR        :  (prune.prune_related_linear, prune.prune_linear),
                OPTYPE.CONCAT        :  (_prune_concat,              _prune_concat),
                OPTYPE.ELEMENTWISE   :  (_prune_elementwise_op,      _prune_elementwise_op),
               }
    OUTPUT_NODE_RULES = {}
    INPUT_NODE_RULES = {}
    for t1 in HANDLER.keys():
        for t2 in HANDLER.keys():
            OUTPUT_NODE_RULES[ (t1, t2) ] = (HANDLER[t1][1], HANDLER[t2][0]) # change in_channels of output layer
            INPUT_NODE_RULES[ (t1, t2) ] = (HANDLER[t1][0], HANDLER[t2][1]) # change out_channels of input layer

    def build_dependency( self, model, example_inputs, get_output_fn=None ):
        # get module name
        self._module_to_name = { module: name for (name, module) in model.named_modules() }
        # build dependency graph
        self.module_to_node = self._obtain_forward_graph( model, example_inputs, get_output_fn=get_output_fn )
        self._build_dependency(self.module_to_node)
        self.update_index()
        return self

    def update_index( self ):
        for module, node in self.module_to_node.items():
            if node.type==OPTYPE.LINEAR:
                self._set_fc_index_transform( node )
            if node.type==OPTYPE.CONCAT:
                self._set_concat_index_transform(node)

    def get_pruning_plan(self, module, pruning_fn, idxs): 
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
                    dep = Dependency( trigger=in_node_rule[0], handler=in_node_rule[1], broken_node=in_node)
                    node.dependencies.append( dep )

            for out_node in node.outputs:
                out_node_rule = self.OUTPUT_NODE_RULES.get( (node.type, out_node.type), None )
                if out_node_rule is not None:
                    dep = Dependency( trigger=out_node_rule[0], handler=out_node_rule[1], broken_node=out_node)
                    node.dependencies.append( dep )
    
    def _obtain_forward_graph(self, model, example_inputs, get_output_fn):
        #module_to_node = { m: Node( m ) for m in model.modules() if isinstance( m, self.PRUNABLE_MODULES ) }
        model.eval().cpu()
        # Get grad_fn from prunable modules
        grad_fn_to_module = {}
        def _record_module_grad_fn(module, inputs, outputs):
            grad_fn_to_module[outputs.grad_fn] = module

        hooks = [m.register_forward_hook(_record_module_grad_fn) for m in model.modules() if isinstance( m, self.PRUNABLE_MODULES ) ]
        out = model(example_inputs)
        for hook in hooks:
            hook.remove()
        grad_fn_from_prunable_module = list( grad_fn_to_module.keys() )

        # create nodes and dummy modules
        visited = set()
        module_to_node = {}

        def _build_graph(grad_fn):
            module = grad_fn_to_module.get( grad_fn, None )
            if module is not None and module in module_to_node:
                return module_to_node[module]
            
            if grad_fn in grad_fn_from_prunable_module: # prunable ops
                module = grad_fn_to_module[ grad_fn ]
            else: # create dummy modules
                if 'catbackward' in grad_fn.name().lower(): # concat op
                    module = _ConcatOp()
                else:
                    module = _ElementWiseOp()   # All other ops are treated as element-wise ops
                grad_fn_to_module[ grad_fn ] = module # record grad_fn

            node = Node( module, grad_fn, self._module_to_name.get( module, None ) )
            module_to_node[ module ] = node

            if hasattr(grad_fn, 'next_functions'):
                for f in grad_fn.next_functions:
                    if f[0] is not None and 'accumulategrad' not in f[0].name().lower():
                        input_node = _build_graph(f[0])
                        # connect nodes
                        node.add_input( input_node )
                        input_node.add_output( node )
            #print(grad_fn, node)
            return node
        
        if get_output_fn is not None:
            out = get_output_fn(out)
            
        if isinstance(out, (list, tuple) ):
            for o in out:
                _build_graph( o.grad_fn )
        else:
            _build_graph( out.grad_fn )
        return module_to_node

    def _set_fc_index_transform(self, fc_node: Node):
        if fc_node.type != OPTYPE.LINEAR:
            return
        
        visited = set()
        fc_in_features = fc_node.module.in_features
        feature_channels = _get_input_channels(fc_node.inputs[0])
        stride = fc_in_features // feature_channels
        if stride>1:
            for in_node in fc_node.inputs:
                for dep in fc_node.dependencies:
                    if dep.broken_node==in_node:
                        dep.index_transform = _StrideIndexTransform( stride=stride, reverse=True )
                
                for dep in in_node.dependencies:
                    if dep.broken_node == fc_node:
                        dep.index_transform = _StrideIndexTransform( stride=stride, reverse=False )

    def _set_concat_index_transform(self, cat_node: Node):
        if cat_node.type != OPTYPE.CONCAT:
            return
        
        chs = []
        for n in cat_node.inputs:
            chs.append( _get_input_channels(n) )

        offsets = [0]
        for ch in chs:
            offsets.append( offsets[-1]+ch )
        cat_node.module.offsets = offsets

        for i, in_node in enumerate(cat_node.inputs):
            for dep in cat_node.dependencies:
                if dep.broken_node == in_node:
                    dep.index_transform = _OffsetIndexTransform( offset=offsets[i:i+2], reverse=True )

            for dep in in_node.dependencies:
                if dep.broken_node == cat_node:
                    dep.index_transform = _OffsetIndexTransform( offset=offsets[i:i+2], reverse=False )

def _get_input_channels(node):
    ch = _get_node_channel(node)
    if ch is None:
        ch = 0
        for in_node in node.inputs:
            if node.type==OPTYPE.CONCAT:
                ch+=_get_input_channels( in_node )
            else:
                ch=_get_input_channels( in_node )
    return ch