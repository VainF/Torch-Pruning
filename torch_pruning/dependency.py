import typing
import warnings
from numbers import Number
import torch
import torch.nn as nn
from .pruner import function
from . import _helpers, utils, ops

__all__ = ["Dependency", "Group", "DependencyGraph"]

INDEX_MAPPING_PLACEHOLDER = None
MAX_RECURSION_DEPTH = 500

class Node(object):
    """ Node of DepGraph
    """
    def __init__(self, module: nn.Module, grad_fn, name: str = None):
        # For Computational Graph (Tracing)
        self.inputs = []  # input nodes
        self.outputs = [] # output nodes
        self.module = module # reference to torch.nn.Module
        self.grad_fn = grad_fn # grad_fn of nn.module output

        self._name = name # node name
        self.type = ops.module2type(module) # node type (enum), op.OPTYPE
        self.module_class = module.__class__ # class type of the module

        # For Dependency Graph
        self.dependencies = []  # Adjacency List
        self.enable_index_mapping = True # whether to enable index mapping
        self.pruning_dim = -1 # the dimension to be pruned

    @property
    def name(self):
        if self._name is None:
            return str(self.module)
        else:
            fmt = self._name
            if self.type != ops.OPTYPE.PARAMETER:
                fmt += " ({})".format(str(self.module))
            return fmt

    def add_input(self, node):
        self.inputs.append(node)


    def add_output(self, node):
        self.outputs.append(node)


    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<Node: ({})>".format(self.name)

    def details(self):
        fmt = "-" * 32 + "\n"
        fmt += "<Node: ({})>\n".format(self.name)
        fmt += " " * 4 + "IN:\n"
        for in_node in self.inputs:
            fmt += " " * 8 + "{}\n".format(in_node)
        fmt += " " * 4 + "OUT:\n"
        for out_node in self.outputs:
            fmt += " " * 8 + "{}\n".format(out_node)
        fmt += " " * 4 + "DEP:\n"
        for dep in self.dependencies:
            fmt += " " * 8 + "{}\n".format(dep)
        fmt += "\tEnable_index_mapping={}, pruning_dim={}\n".format(
            self.enable_index_mapping, self.pruning_dim)
        fmt = "-" * 32 + "\n"
        return fmt


class Edge(): # for readability
    pass


class Dependency(Edge):
    def __init__(
        self,
        trigger: typing.Callable,
        handler: typing.Callable,
        source: Node,
        target: Node,
    ):
        """Layer dependency (Edge of DepGraph).
        Args:
            trigger (Callable): a pruning function that triggers this dependency
            handler (Callable): a pruning function that can fix the broken dependency
            source (Node): the source node pruned by the trigger function
            target (Node): the target node pruned by the handler function
            index_mapping (Callable): a callable function for index mapping
        """
        self.trigger = trigger
        self.handler = handler
        self.source = source
        self.target = target             
        # Current coordinate system => Standard coordinate system => target coordinate system 
        #                     index_mapping[0]              index_mapping[1]
        self.index_mapping = [INDEX_MAPPING_PLACEHOLDER, INDEX_MAPPING_PLACEHOLDER] # [None, None] by default

    def __call__(self, idxs: list):
        self.handler.__self__.pruning_dim = self.target.pruning_dim # set pruning_dim
        if len(idxs)>0 and isinstance(idxs[0], _helpers._HybridIndex):
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
        return pruning_fn == self.trigger

    def __eq__(self, other):
        return (
            self.source == other.source 
            and self.trigger == other.trigger
            and self.handler == other.handler
            and self.target == other.target
        )
    
    @property
    def layer(self):
        return self.target.module

    @property
    def pruning_fn(self):
        return self.handler

    def __hash__(self):
        return hash((self.source, self.target, self.trigger, self.handler))


class Group(object):
    """A group that contains dependencies and pruning indices.   
    Each element is defined as as namedtuple('_helpers.GroupItem'. 

    group := [ (Dep1, Indices1), (Dep2, Indices2), ..., (DepK, IndicesK) ]
    """

    def __init__(self):
        self._group = list()
        self._DG = None # the dependency graph that this group belongs to

    def prune(self, idxs=None, record_history=True):
        """Prune all coupled layers in the group
        """
        if idxs is not None: # prune the group with user-specified indices
            module = self._group[0].dep.target.module
            pruning_fn = self._group[0].dep.handler
            new_group = self._DG.get_pruning_group(module, pruning_fn, idxs) # create a new group with the specified indices
            new_group.prune()
        else:
            for dep, idxs in self._group:
                if dep.target.type == ops.OPTYPE.PARAMETER: 
                    # prune unwrapped nn.Parameter
                    old_parameter = dep.target.module
                    name = self._DG._param_to_name[old_parameter]
                    self._DG._param_to_name.pop(old_parameter)
                    pruned_parameter = dep(idxs)
                    path = name.split('.')
                    module = self._DG.model
                    for p in path[:-1]:
                        module = getattr(module, p)
                    setattr(module, path[-1], pruned_parameter)
                    self._DG._param_to_name[pruned_parameter] = name
                    self._DG.module2node[pruned_parameter] = self._DG.module2node.pop(old_parameter)
                    self._DG.module2node[pruned_parameter].module = pruned_parameter           
                else:
                    dep(idxs)
        
        if record_history:
            root_module, pruning_fn, root_pruning_idx = self[0][0].target.module, self[0][0].trigger, self[0][1]
            root_module_name = self._DG._module2name[root_module]
            self._DG._pruning_history.append([root_module_name, self._DG.is_out_channel_pruning_fn(pruning_fn), root_pruning_idx])
    
    def add_dep(self, dep, idxs):
        self._group.append(_helpers.GroupItem(dep=dep, idxs=idxs))

    def __getitem__(self, k):
        return self._group[k]

    def __setitem__(self, k, v):
        self._group[k] = v

    @property
    def items(self):
        return self._group

    def has_dep(self, dep):
        for _dep, _ in self._group:
            if dep == _dep:
                return True
        return False

    def has_pruning_op(self, dep: Dependency, idxs: _helpers._HybridIndex):
        for _dep, _idxs in self._group:
            #_idxs = _helpers.to_plain_idxs(_idxs)
            if (
                _dep.target == dep.target
                and _dep.handler == dep.handler
                and _idxs == idxs
            ):
                return True
        return False

    def __len__(self):
        return len(self._group)

    def add_and_merge(self, dep, idxs):
        for i, (_dep, _idxs) in enumerate(self._group):
            if _dep.target == dep.target and _dep.handler == dep.handler:
                visited_idxs = set()
                merged_idxs = []
                for index in _idxs + idxs:
                    if index.idx not in visited_idxs:
                        merged_idxs.append(index)
                        visited_idxs.add(index.idx)
                self._group[i] = _helpers.GroupItem(dep=_dep, idxs=merged_idxs)
                return
        self.add_dep(dep, idxs)

    def __str__(self):
        fmt = ""
        fmt += "\n" + "-" * 32 + "\n"
        fmt += " " * 10 + "Pruning Group"
        fmt += "\n" + "-" * 32 + "\n"
        for i, (dep, idxs) in enumerate(self._group):
            fmt += "[{}] {}, #idxs={}\n".format(i, dep, len(idxs))
        fmt += "-" * 32 + "\n"
        return fmt

    def details(self):
        fmt = ""
        fmt += "\n" + "-" * 32 + "\n"
        fmt += " " * 10 + "Pruning Group"
        fmt += "\n" + "-" * 32 + "\n"
        for i, (dep, idxs) in enumerate(self._group):
            if i==0: 
                fmt += "[{}] {}, idxs ({}) ={}  (Pruning Root)\n".format(i, dep, len(idxs), idxs)
            else:
                fmt += "[{}] {}, idxs ({}) ={} \n".format(i, dep, len(idxs), idxs)
        fmt += "-" * 32 + "\n"
        return fmt

    def exec(self):
        """old interface, will be deprecated in the future."""
        warnings.warn("Group.exec() will be deprecated in the future. Please use Group.prune() instead.")
        self.prune()

    def __call__(self):
        return self.prune()


class DependencyGraph(object):

    def __init__(self):
        _dummy_pruners = {
            ops.OPTYPE.CONCAT: ops.ConcatPruner(),
            ops.OPTYPE.SPLIT: ops.SplitPruner(),
            ops.OPTYPE.ELEMENTWISE: ops.ElementWisePruner(),
            ops.OPTYPE.RESHAPE: ops.ReshapePruner(),
            ops.OPTYPE.UNBIND: ops.UnbindPruner(),
            ops.OPTYPE.CUSTOMIZED: ops.CustomizedPruner(), # just a placeholder
        }
        self.REGISTERED_PRUNERS = function.PrunerBox.copy()  # shallow copy
        self.REGISTERED_PRUNERS.update(_dummy_pruners) # merge dummy pruners
        self.CUSTOMIZED_PRUNERS = {} # user-customized pruners

        self.IGNORED_LAYERS_IN_TRACING = []

        # cache pruning functions for fast lookup
        self._in_channel_pruning_fn = set([p.prune_in_channels for p in self.REGISTERED_PRUNERS.values() if p is not None] + [p.prune_in_channels for p in self.CUSTOMIZED_PRUNERS.values() if p is not None])
        self._out_channel_pruning_fn = set([p.prune_out_channels for p in self.REGISTERED_PRUNERS.values() if p is not None] + [p.prune_out_channels for p in self.CUSTOMIZED_PRUNERS.values() if p is not None])
        self._op_id = 0 # operatior id, will be increased by 1 for each new operator

        # Pruning History
        self._pruning_history = []

    def pruning_history(self) -> typing.List[typing.Tuple[str, bool, typing.Union[list, tuple]]]:
        return self._pruning_history

    def load_pruning_history(self, pruning_history):
        """Redo the pruning history"""
        self._pruning_history = pruning_history
        for module_name, is_out_channel_pruning, pruning_idx in self._pruning_history:
            module = self.model
            for n in module_name.split('.'):
                module = getattr(module, n)
            pruner = self.get_pruner_of_module(module)
            if is_out_channel_pruning:
                pruning_fn = pruner.prune_out_channels
            else:
                pruning_fn = pruner.prune_in_channels
            group = self.get_pruning_group(module, pruning_fn, pruning_idx)
            group.prune(record_history=False)
            
    def build_dependency(
        self,
        model: torch.nn.Module,
        example_inputs: typing.Union[torch.Tensor, typing.Sequence, typing.Dict],
        forward_fn: typing.Callable[[torch.nn.Module, typing.Union[torch.Tensor, typing.Sequence]], torch.Tensor] = None,
        output_transform: typing.Callable = None,
        unwrapped_parameters: typing.Dict[nn.Parameter, int] = None,
        customized_pruners: typing.Dict[ typing.Union[typing.Any, torch.nn.Module],function.BasePruningFunc] = None,
        ignored_layers: typing.List[nn.Module] = None,
        ignored_params: typing.List[nn.Parameter] = None,
        verbose: bool = True,
    ) -> "DependencyGraph":
        """Build a dependency graph through tracing.
        Args:
            model (class): the model to be pruned.
            example_inputs (torch.Tensor or List): dummy inputs for tracing.
            forward_fn (Callable): a function to forward the model with example_inputs, which should return a reduced scalr tensor for backpropagation.
            output_transform (Callable): a function to transform network outputs.
            unwrapped_parameters (typing.Dict[nn.Parameter, int]): unwrapped nn.parameters that do not belong to standard nn.Module.
            customized_pruners (typing.Dict[ typing.Union[typing.Any, torch.nn.Module],function.BasePruningFunc]): customized pruners for a specific layer type or a specific layer instance.
            ignored_layers (typing.List[nn.Module]): ignored layers that will not be traced in the dependency graph.
            ignored_params (typing.List[nn.Parameter]): ignored nn.Parameter that will not be pruned.
            verbose (bool): verbose mode.
        """

        self.verbose = verbose
        self.model = model
        self._module2name = {module: name for (name, module) in model.named_modules()} # nn.Module => module name

        # Register customized pruners
        if customized_pruners is not None:
            for customized_type, customized_pruner in customized_pruners.items():
                self.register_customized_layer(customized_type, customized_pruner)
        
        if ignored_layers is not None:
            self.IGNORED_LAYERS_IN_TRACING.extend(ignored_layers)
        self.ignored_params = ignored_params
        # Ignore all sub-modules of customized layers since they will be handled by the customized pruner
        for layer_type_or_instance in self.CUSTOMIZED_PRUNERS.keys():            
            for m in self.model.modules():
                # a layer instance or a layer type
                if (m==layer_type_or_instance) or (not isinstance(layer_type_or_instance, torch.nn.Module) and isinstance(m, layer_type_or_instance)):
                    for sub_module in m.modules(): 
                        if sub_module != m:
                            self.IGNORED_LAYERS_IN_TRACING.append(sub_module)

        # Detect unwrapped nn.parameters that can not be handled by self.REGISTED_PRUNERS
        self._param_to_name, self.unwrapped_parameters = self._detect_unwrapped_parameters(unwrapped_parameters)

        # Detect torch.no_grad()
        assert torch.is_grad_enabled(), "Dependency graph relies on autograd for tracing. Please check and disable the torch.no_grad() in your code."
        
        # Build computational graph through tracing. 
        self.module2node = self._trace(
            model, example_inputs, forward_fn, output_transform=output_transform
        )

        # Build dependency graph
        self._build_dependency(self.module2node)
        
        # Initialize shape information
        self._init_shape_information()

        # Update index mapping for torch.cat/split/chunck/...
        self.update_index_mapping()
        return self

    def register_customized_layer(
        self,
        layer_type_or_instance: typing.Union[typing.Any, torch.nn.Module],
        layer_pruner: function.BasePruningFunc,
    ):
        """Register a customized pruner
        Args:
            layer_type (class): the type of target layer
            pruner (tp.pruner.BasePruningFunc): a pruner for the specified layer type.
        """
        self.CUSTOMIZED_PRUNERS[layer_type_or_instance] = layer_pruner
        
        # Update cache
        self._in_channel_pruning_fn = set([p.prune_in_channels for p in self.REGISTERED_PRUNERS.values() if p is not None] + [p.prune_in_channels for p in self.CUSTOMIZED_PRUNERS.values() if p is not None])
        self._out_channel_pruning_fn = set([p.prune_out_channels for p in self.REGISTERED_PRUNERS.values() if p is not None] + [p.prune_out_channels for p in self.CUSTOMIZED_PRUNERS.values() if p is not None])

    def check_pruning_group(self, group: Group) -> bool:
        """check the group to avoid over-pruning. Return True if there are sufficient prunable elements.
        Args:
            group (Group): a depenedency group
        """
        for dep, idxs in group:
            if self.is_out_channel_pruning_fn(dep.handler):
                prunable_chs = self.get_out_channels(
                    dep.target.module)
                if prunable_chs is None: continue
                if prunable_chs <= len(idxs):
                    return False

            if self.is_in_channel_pruning_fn(dep.handler):
                prunable_in_chs = self.get_in_channels(
                    dep.target.module)
                if prunable_in_chs is None: continue
                if prunable_in_chs <= len(idxs):
                    return False
        return True

    def is_out_channel_pruning_fn(self, fn: typing.Callable) -> bool:
        return (fn in self._out_channel_pruning_fn)
    
    def is_in_channel_pruning_fn(self, fn: typing.Callable) -> bool:
        return (fn in self._in_channel_pruning_fn)

    def get_pruning_group(
        self,
        module: nn.Module,
        pruning_fn: typing.Callable,
        idxs: typing.Sequence[int],
    ) -> Group:
        """
        Get the pruning group for a given module.

            Args:
                module (nn.Module): The module to be pruned.
                pruning_fn (Callable): The pruning function.
                idxs (list or tuple): The indices of channels/dimensions.

            Returns:
                Group: The pruning group containing the dependencies and indices.

            Raises:
                ValueError: If the module is not in the dependency graph.
        """
        if module not in self.module2node:
            raise ValueError(
                "Module {} is not in the dependency graph.".format(module)
            )
        if isinstance(module, ops.TORCH_CONV) and module.groups == module.out_channels and module.out_channels>1:
            pruning_fn = function.prune_depthwise_conv_out_channels
        if isinstance(idxs, Number):
            idxs = [idxs]
        
        idxs = [ _helpers._HybridIndex(idx=i, root_idx=i) for i in idxs ] # idxs == root_idxs for the root layer

        self.update_index_mapping()
        group = Group()

        #  the user pruning operation
        root_node = self.module2node[module]
        group.add_dep(
            dep=Dependency(pruning_fn, pruning_fn, source=root_node, target=root_node), 
            idxs=idxs,
        )

        visited_node = set()

        def _fix_dependency_graph_non_recursive(dep, idxs, *args):
            processing_stack = [(dep, idxs)]
            while len(processing_stack) > 0:
                dep, idxs = processing_stack.pop(-1)
                node, fn = dep.target, dep.handler
                visited_node.add(node)
    
                for new_dep in node.dependencies:
                    if new_dep.is_triggered_by(fn):
                        new_indices = idxs
                        for mapping in new_dep.index_mapping:
                            if mapping is not None:
                                new_indices = mapping(new_indices)

                        if len(new_indices) == 0:
                            continue
                        if (new_dep.target in visited_node) and group.has_pruning_op(
                            new_dep, new_indices
                        ):
                            continue
                        else:
                            group.add_dep(new_dep, new_indices)
                            processing_stack.append(
                                (new_dep, new_indices)
                            )

        _fix_dependency_graph_non_recursive(*group[0])
        # merge pruning ops
        merged_group = Group()
        for dep, idxs in group.items:
            if isinstance(dep.target.module, nn.Parameter): #and dep.target.module in self.ignored_params:
                skip=False
                for ignored_p in self.ignored_params:
                    if dep.target.module is ignored_p:
                        skip=True
                        break
                if skip:
                    continue
            merged_group.add_and_merge(dep, idxs)
        merged_group._DG = self
        for i in range(len(merged_group)):
            hybrid_idxs = merged_group[i].idxs
            idxs = _helpers.to_plain_idxs(hybrid_idxs)
            root_idxs = _helpers.to_root_idxs(hybrid_idxs)
            merged_group[i] = _helpers.GroupItem(merged_group[i].dep, idxs) # transform _helpers._HybridIndex to plain index
            merged_group[i].root_idxs = root_idxs
        return merged_group

    def get_all_groups(self, ignored_layers=[], root_module_types=(ops.TORCH_CONV, ops.TORCH_LINEAR)):
        """
            Get all pruning groups for the given module. Groups are generated on the module typs specified in root_module_types.

            Args:
                ignored_layers (list): List of layers to be ignored during pruning.
                root_module_types (tuple): Tuple of root module types to consider for pruning.

            Yields:
                list: A pruning group containing dependencies and their corresponding pruning handlers.

            Example:
            ```python
            for group in DG.get_all_groups(ignored_layers=[layer1, layer2], root_module_types=[nn.Conv2d]):
                print(group)
            ```
        """
        visited_layers = []
        ignored_layers = ignored_layers+self.IGNORED_LAYERS_IN_TRACING

        for m in list(self.module2node.keys()):
            
            if m in ignored_layers:
                continue
            
            if not isinstance(m, tuple(root_module_types)):
                continue

            pruner = self.get_pruner_of_module(m)
            if pruner is None or pruner.get_out_channels(m) is None:
                continue

            if m in visited_layers:
                continue

            layer_channels = pruner.get_out_channels(m)
            group = self.get_pruning_group(
                m, pruner.prune_out_channels, list(range(layer_channels)))
            
            prunable_group = True
            for dep, _ in group:
                module = dep.target.module
                pruning_fn = dep.handler
                if self.is_out_channel_pruning_fn(pruning_fn):
                    visited_layers.append(module)
                    if module in ignored_layers:
                        prunable_group = False
            if prunable_group:
                yield group

    def get_pruner_of_module(self, module: nn.Module):
        p = self.CUSTOMIZED_PRUNERS.get(module.__class__, None) # customized pruners for a specific layer type
        if p is None:
            p = self.REGISTERED_PRUNERS.get(ops.module2type(module), None) # standard pruners
        return p

    def get_out_channels(self, module_or_node):
        if isinstance(module_or_node, Node):
            module = module_or_node.module
            pruning_dim = module_or_node.pruning_dim
        else:
            module = module_or_node
            pruning_dim = self.module2node[module].pruning_dim
        p = self.get_pruner_of_module(module)
        p.pruning_dim = pruning_dim
        if p is None:
            return None
        return p.get_out_channels(module)

    def get_in_channels(self, module_or_node):
        if isinstance(module_or_node, Node):
            module = module_or_node.module
            pruning_dim = module_or_node.pruning_dim
        else:
            module = module_or_node
            pruning_dim = self.module2node[module].pruning_dim
        p = self.get_pruner_of_module(module)
        p.pruning_dim = pruning_dim
        if p is None:
            return None
        return p.get_in_channels(module)

    def _infer_out_channels_recursively(self, node: Node, recursive_depth: list):
        """ infer the number of output channels recursively
        """     
        if recursive_depth[0] > MAX_RECURSION_DEPTH:
            warnings.warn("Maximum recursive depth reached!")
            return None
        ch = self.get_out_channels(node)
        if ch is None:
            ch = 0
            for in_node in node.inputs:
                if node.type == ops.OPTYPE.CONCAT:
                    recursive_depth[0]+=1
                    sub_ch = self._infer_out_channels_recursively(in_node, recursive_depth)
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
                        ch = self._infer_out_channels_recursively(in_node, recursive_depth)
            if ch == 0:
                return None
        return ch

    def _infer_in_channels_recursively(self, node: Node, recursive_depth: list):
        """ infer the number of input channels recursively
        """         
        if recursive_depth[0] > MAX_RECURSION_DEPTH:
            return None
        ch = self.get_in_channels(node)
        if ch is None:
            ch = 0
            for out_node in node.outputs:
                if node.type == ops.OPTYPE.SPLIT:
                    recursive_depth[0]+=1
                    sub_ch = self._infer_in_channels_recursively(out_node, recursive_depth)
                    if sub_ch is None:
                        return None
                    ch += sub_ch
                else:
                    recursive_depth[0]+=1
                    ch = self._infer_in_channels_recursively(out_node, recursive_depth)
            if ch == 0:
                return None
        return ch
    
    def _detect_unwrapped_parameters(self, unwrapped_parameters):
        # Detect wrapped nn.Parameters
        wrapped_parameters = []
        prunable_module_types = self.REGISTERED_PRUNERS.keys()
        for m in self.model.modules():
            op_type = ops.module2type(m)
            if ( op_type in prunable_module_types and op_type!=ops.OPTYPE.ELEMENTWISE ) or m.__class__ in self.CUSTOMIZED_PRUNERS.keys() or m in self.CUSTOMIZED_PRUNERS.keys():
                wrapped_parameters.extend(list(m.parameters()))
       
        # Detect unwrapped nn.Parameters
        unwrapped_detected = []
        _param_to_name = {}
        for name, p in self.model.named_parameters():
            is_wrapped = False
            for p_wrapped in wrapped_parameters:
                if p is p_wrapped:
                    is_wrapped = True
                    break
            if not is_wrapped:
                unwrapped_detected.append(p)
                _param_to_name[p] = name
        if unwrapped_parameters is None:
            unwrapped_parameters = []
        unwrapped_detected = list( set(unwrapped_detected) - set([p for (p, _) in unwrapped_parameters]) )
        if len(unwrapped_detected)>0 and self.verbose:
            warning_str = "Unwrapped parameters detected: {}.\n Torch-Pruning will prune the last non-singleton dimension of these parameters. If you wish to change this behavior, please provide an unwrapped_parameters argument.".format([_param_to_name[p] for p in unwrapped_detected])
            warnings.warn(warning_str)
        
        # set default pruning dim for unwrapped parameters
        for p in unwrapped_detected:
            # get the last dimension that >1
            def last_non_singleton_dim(tensor):
                non_singleton_dims = [i for i, s in enumerate(tensor.shape) if s > 1]
                return non_singleton_dims[-1] if non_singleton_dims else None
            pruning_dim = last_non_singleton_dim(p)
            if pruning_dim is not None:
                unwrapped_parameters.append( _helpers.UnwrappedParameters(parameters=p, pruning_dim=pruning_dim) ) # prune the last non-singleton dim by daufault
        return _param_to_name, unwrapped_parameters
    
    def _build_dependency(self, module2node):

        for _, node in module2node.items():
            ###########################################
            # Rule 1) - Inter-layer Dependency
            ###########################################
            for in_node in node.inputs:
                handler = self.get_pruner_of_module(in_node.module).prune_out_channels
                trigger = self.get_pruner_of_module(node.module).prune_in_channels
                dep = Dependency(
                    trigger=trigger, handler=handler, source=node, target=in_node
                )
                node.dependencies.append(dep)

            for out_node in node.outputs:
                trigger = self.get_pruner_of_module(node.module).prune_out_channels
                handler = self.get_pruner_of_module(out_node.module).prune_in_channels
                dep = Dependency(
                    trigger=trigger, handler=handler, source=node, target=out_node
                )
                node.dependencies.append(dep)

            ###########################################
            # Rule 2) - Intra-layer Dependency
            ###########################################

            # This is implictly implemented by assigning
            # prune_out_channels=prune_in_channels in tp.pruner.function.BasePruningFunc

    def _trace(self, model, example_inputs, forward_fn, output_transform):
        """ Tracing the model as a graph
        """
        model.eval()
        gradfn2module = {}
        visited = {}
        self._2d_4d = True # only for pytorch<=1.8
        def _record_grad_fn(module, inputs, outputs):
            
            if module not in visited:
                visited[module] = 1
            else:
                visited[module] += 1
            
            if isinstance(module, nn.Linear) and len(outputs.shape)==3:
                self._2d_4d=False

            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if isinstance(outputs, torch.nn.utils.rnn.PackedSequence):
                outputs = outputs.data

            gradfn2module[outputs.grad_fn] = module
            

        # Register hooks for prunable modules
        registered_types = tuple(ops.type2class(
            t) for t in self.REGISTERED_PRUNERS.keys()) + tuple(self.CUSTOMIZED_PRUNERS.keys())
        hooks = [
            m.register_forward_hook(_record_grad_fn)
            for m in model.modules()
            if (isinstance(m, registered_types) and m not in self.IGNORED_LAYERS_IN_TRACING)
        ]

        # Feed forward to record gradient functions of prunable modules
        if forward_fn is not None:
            out = forward_fn(model, example_inputs)
        elif isinstance(example_inputs, dict):
            out = model(**example_inputs)
        else:
            try:
                out = model(*example_inputs)
            except:
                out = model(example_inputs)
        for hook in hooks:
            hook.remove()

        # for recursive models or layers
        reused = [m for (m, count) in visited.items() if count > 1]
        if output_transform is not None:
            out = output_transform(out)

        # Graph tracing
        module2node = {} # create a mapping from nn.Module to tp.dependency.Node
        visited = set()
        for o in utils.flatten_as_list(out):
            self._trace_computational_graph(
                module2node, o.grad_fn, gradfn2module, reused, visited=visited)

        # TODO: Improving ViT pruning
        # This is a corner case for pruning ViT,
        # where concatination of pos_emb and cls_emv is not applied on the feature dim.
        # Notably, this is not a good practice and will be fixed in the future version
        if len(self.unwrapped_parameters) > 0:
            for node in module2node.values():
                if node.type in (ops.OPTYPE.CONCAT, ops.OPTYPE.SPLIT):
                    stack = [node]
                    visited = set()
                    while len(stack) > 0:
                        n = stack.pop(-1)
                        visited.add(n)
                        if n.type == ops.OPTYPE.PARAMETER and len(n.module.shape) == 3:
                            node.enable_index_mapping = False
                            break
                        else:
                            for ni in n.inputs:
                                if ni not in visited:
                                    stack.append(ni)
        return module2node

    def _trace_computational_graph(self, module2node, grad_fn_root, gradfn2module, reused, visited=set()):

        def create_node_if_not_exists(grad_fn):
            module = gradfn2module.get(grad_fn, None)
            if module is not None and module in module2node and module not in reused:
                return module2node[module]

            # 1. link grad_fns and modules
            if module is None:  # a new module
                if not hasattr(grad_fn, "name"):
                    # we treat all unknwon modules as element-wise operations by default,
                    # which does not modify the #dimension/#channel of features.
                    # If you have some customized layers, please register it with DependencyGraph.register_customized_layer
                    module = ops._ElementWiseOp(self._op_id ,"Unknown")
                    self._op_id+=1
                    if self.verbose:
                        warnings.warn(
                            "[Warning] Unknown operation {} encountered, which will be handled as an element-wise op".format(
                                str(grad_fn))
                        )
                elif "catbackward" in grad_fn.name().lower():
                    module = ops._ConcatOp(self._op_id)
                    self._op_id+=1
                elif "split" in grad_fn.name().lower():
                    module = ops._SplitOp(self._op_id)
                    self._op_id+=1
                elif "unbind" in grad_fn.name().lower():
                    module = ops._UnbindOp(self._op_id)
                    self._op_id+=1
                elif "view" in grad_fn.name().lower() or 'reshape' in grad_fn.name().lower():
                    module = ops._ReshapeOp(self._op_id)
                    self._op_id+=1
                else:
                    # treate other ops as element-wise ones, like Add, Sub, Div, Mul.
                    module = ops._ElementWiseOp(self._op_id, grad_fn.name())
                    self._op_id+=1
                gradfn2module[grad_fn] = module

            # 2. link modules and nodes
            if module not in module2node:
                node = Node(
                    module=module,
                    grad_fn=grad_fn,
                    name=self._module2name.get(module, None),
                )
                if (
                    type(module) in self.CUSTOMIZED_PRUNERS
                ):  # mark it as a customized layer
                    node.type = ops.OPTYPE.CUSTOMIZED
                module2node[module] = node
            else:
                node = module2node[module]
            return node

        # non-recursive construction of computational graph
        processing_stack = [grad_fn_root]
        while len(processing_stack) > 0:
            grad_fn = processing_stack.pop(-1)
            if grad_fn in visited:
                continue
            node = create_node_if_not_exists(grad_fn=grad_fn)
            if hasattr(grad_fn, "next_functions"):
                for f in grad_fn.next_functions:
                    if f[0] is not None:
                        if (
                            hasattr(f[0], "name")
                            and "accumulategrad" in f[0].name().lower()
                        ):  # a leaf variable.
                            is_unwrapped_param = False
                            for (j, (p, dim)) in enumerate(self.unwrapped_parameters):
                                if f[0].variable is p:
                                    is_unwrapped_param = True
                                    gradfn2module[f[0]] = p
                                    self._module2name[p] = "UnwrappedParameter_{} ({})".format(j, p.shape)
                            if not is_unwrapped_param:
                                continue
                        input_node = create_node_if_not_exists(f[0])
                        node.add_input(input_node)
                        input_node.add_output(node)
                        processing_stack.append(f[0])
            visited.add(grad_fn)

        
        for (param, dim) in self.unwrapped_parameters:
            if param in module2node:
                module2node[param].pruning_dim = dim
        return module2node

    def update_index_mapping(self):
        """ update all index mapping after pruning
        """       
        # update index mapping
        for module, node in self.module2node.items():
            if node.type == ops.OPTYPE.CONCAT:
                self._update_concat_index_mapping(node)
            if node.type == ops.OPTYPE.SPLIT:
                self._update_split_index_mapping(node)
            if node.type == ops.OPTYPE.RESHAPE:
                self._update_reshape_index_mapping(node)
            if node.type == ops.OPTYPE.UNBIND:
                self._update_unbind_index_mapping(node)

    def _init_shape_information(self):
        for module, node in self.module2node.items():

            if node.type == ops.OPTYPE.SPLIT:
                grad_fn = node.grad_fn

                if hasattr(grad_fn, '_saved_self_sizes') or hasattr(grad_fn, '_saved_split_sizes'):
                    MAX_LEGAL_DIM = 100
                    
                    if hasattr(grad_fn, '_saved_split_sizes') and hasattr(grad_fn, '_saved_dim') :
                        if grad_fn._saved_dim != 1 and grad_fn._saved_dim < MAX_LEGAL_DIM: # a temp fix for pytorch==1.11, where the _saved_dim is an uninitialized value like 118745347895359
                            continue
                        chs = list(grad_fn._saved_split_sizes)
                        node.module.split_sizes = chs
                    elif hasattr(grad_fn, '_saved_split_size') and hasattr(grad_fn, '_saved_dim'):
                        if grad_fn._saved_dim != 1 and grad_fn._saved_dim < MAX_LEGAL_DIM: # a temp fix for pytorch==1.11, where the _saved_dim is an uninitialized value like 118745347895359
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
                        chs.append(self._infer_in_channels_recursively(n, recursive_depth))
                    offsets = [0]
                    for ch in chs:
                        if ch is None: continue
                        offsets.append(offsets[-1] + ch)
                    node.module.split_sizes = chs
                    node.module.offsets = offsets
                                                
    def _update_flatten_index_mapping(self, fc_node: Node):
        if fc_node.type != ops.OPTYPE.LINEAR:
            return
        fc_in_features = fc_node.module.in_features
        feature_channels = 0
        for n in fc_node.inputs:
            recursive_depth = [0]
            feature_channels = self._infer_out_channels_recursively(n, recursive_depth)
            if feature_channels is not None:  # =0 if there is a residual connection to model inputs
                break
        if (
            feature_channels is None
        ):  # the first layer: https://github.com/VainF/Torch-Pruning/issues/21
            return
        stride = fc_in_features // feature_channels
        if stride > 1 and fc_in_features % feature_channels == 0:
            for in_node in fc_node.inputs:
                for dep in fc_node.dependencies:
                    if dep.target == in_node:
                        dep.index_mapping[0] = _helpers._FlattenIndexMapping(
                            stride=stride, reverse=True
                        )
                for dep in in_node.dependencies:
                    if dep.target == fc_node:
                        dep.index_mapping[0] = _helpers._FlattenIndexMapping(
                            stride=stride, reverse=False
                        )

    def _update_reshape_index_mapping(self, reshape_node: Node):
        
        # Only Supports 2D/4D tensors
        # TODO: Better support for reshape/view/flatten
        if hasattr(reshape_node.grad_fn, '_saved_self_sizes'): 
            size = reshape_node.grad_fn._saved_self_sizes
            if (len(size)!=1 and len(size)!=4):
                return
        else: # old pytorch versions
            if not self._2d_4d:
                return 

        out_channels = None
        for n in reshape_node.outputs:
            recursive_depth = [0]
            out_channels = self._infer_in_channels_recursively(n, recursive_depth)
            if recursive_depth[0] > MAX_RECURSION_DEPTH:
                return
            if out_channels is not None:  # =0 if there is a residual connection to model inputs
                break
        
        in_channels = None
        for n in reshape_node.inputs:
            recursive_depth = [0]
            in_channels = self._infer_out_channels_recursively(n, recursive_depth)
            if recursive_depth[0] > MAX_RECURSION_DEPTH:
                return
            if in_channels is not None:  # =0 if there is a residual connection to model inputs
                break
        
        if out_channels is None or in_channels is None: return
        if out_channels==in_channels: return
        
        if hasattr(reshape_node.grad_fn, '_saved_self_sizes'):
            if len(size)==4 and size[1]*size[2]*size[3]!=out_channels:
                return
        
        # Flatten
        #print(reshape_node.grad_fn._saved_self_sizes, in_channels, out_channels)
        if out_channels > in_channels:
             for in_node in reshape_node.inputs:
                for dep in reshape_node.dependencies:
                    if dep.target == in_node:
                        dep.index_mapping[0] = _helpers._FlattenIndexMapping(
                            stride=out_channels // in_channels, reverse=True
                        )

                for dep in in_node.dependencies:
                    if dep.target == reshape_node:
                        dep.index_mapping[0] = _helpers._FlattenIndexMapping(
                            stride=out_channels // in_channels, reverse=False
                        )
        else: # 1D -> 2D
            for out_node in reshape_node.outputs:
                for dep in reshape_node.dependencies:
                    if dep.target == out_node:
                        dep.index_mapping[0] = _helpers._FlattenIndexMapping(
                            stride=in_channels // out_channels, reverse=True
                        )

                for dep in out_node.dependencies:
                    if dep.target == reshape_node:
                        dep.index_mapping[0] = _helpers._FlattenIndexMapping(
                            stride=in_channels // out_channels, reverse=False
                        )
        #print(in_channels, out_channels)
        #print(reshape_node.grad_fn._saved_self_sizes)
        #print('------')
        
    def _update_concat_index_mapping(self, cat_node: Node):
        if cat_node.type != ops.OPTYPE.CONCAT:
            return

        if hasattr(cat_node.grad_fn, '_saved_dim') and cat_node.grad_fn._saved_dim != 1: # this only works for Pytorch>=1.12
            return 

        if cat_node.module.concat_sizes is not None:
            chs = cat_node.module.concat_sizes
        else:
            chs = []
            for n in cat_node.inputs:
                chs.append(self.infer_channels_between(n, cat_node))
            cat_node.module.concat_sizes = chs

            
        offsets = [0]
        for ch in chs:
            if ch is None: 
                #warnings.warn("Fails to trace the concat operation. It may lead to unexpected results.")
                return
            offsets.append(offsets[-1] + ch)
        cat_node.module.offsets = offsets

        # no transform if the concat dim is different from the feature dim
        # TODO: make the messy for loop more efficient
        addressed_dep = []
        for i, in_node in enumerate(cat_node.inputs):
            for dep in cat_node.dependencies:
                if any((dep is d) for d in addressed_dep): continue
                if dep.target == in_node:
                    if cat_node.enable_index_mapping:
                        dep.index_mapping[1] = _helpers._ConcatIndexMapping(
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
                        dep.index_mapping[1] = _helpers._ConcatIndexMapping(
                            offset=offsets[i: i + 2], reverse=False
                        )
                        addressed_dep.append(dep)
                        break
    
        
    def _update_split_index_mapping(self, split_node: Node):
        if split_node.type != ops.OPTYPE.SPLIT:
            return
        
        if hasattr(split_node.grad_fn, '_saved_dim') and split_node.grad_fn._saved_dim != 1: # this only works for Pytorch>=1.12
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
                        dep.index_mapping[0] = (_helpers._SplitIndexMapping(
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
                        dep.index_mapping[0] = (_helpers._SplitIndexMapping(
                            offset=offsets[i: i + 2], reverse=True
                        ))
                        addressed_dep.append(dep)
                        break

    def _update_unbind_index_mapping(self, unbind_node: Node):
        if unbind_node.type != ops.OPTYPE.UNBIND:
            return

        if hasattr(unbind_node.grad_fn, '_saved_dim') and unbind_node.grad_fn._saved_dim != 0: # For timm attention
            return 

        num_chunks = len(unbind_node.outputs)

        for input_node in unbind_node.inputs:
            input_dims = self._infer_out_channels_recursively(input_node, [0])
            if input_dims is not None:
                break
        if input_dims is None: return
        unbind_node.module.offsets = [i*input_dims//num_chunks for i in range(num_chunks+1)]

        #print(input_dims, num_chunks)
        offsets = unbind_node.module.offsets
        if offsets is None:
            return
        addressed_dep = []
        for i, out_node in enumerate(unbind_node.outputs):
            for dep in unbind_node.dependencies:
                if any((dep is d) for d in addressed_dep): continue
                if dep.target == out_node:
                    if unbind_node.enable_index_mapping:
                        dep.index_mapping[0] = (_helpers._SplitIndexMapping(
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
                        dep.index_mapping[0] = (_helpers._SplitIndexMapping(
                            offset=offsets[i: i + 2], reverse=True
                        ))
                        addressed_dep.append(dep)
                        break

    def infer_channels_between(self, node_1, node_2):
        if node_1.type == ops.OPTYPE.SPLIT:
            for i, n in enumerate(node_1.outputs):
                if n == node_2:
                    return node_1.module.split_sizes[i]
        recursive_depth = [0]
        return self._infer_out_channels_recursively(node_1, recursive_depth)

