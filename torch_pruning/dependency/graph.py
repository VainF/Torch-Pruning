"""DependencyGraph class for modeling and managing pruning dependencies."""

import typing
import warnings
from numbers import Number

import torch
import torch.nn as nn

from .. import _helpers, ops, utils
from ..pruner import function
from . import index_mapping, shape_infer
from .constants import MAX_RECURSION_DEPTH, MAX_VALID_DIM
from .dependency import Dependency
from .group import Group
from .node import Node

class DependencyGraph(object):

    def __init__(self):
        _dummy_pruners = {
            ops.OPTYPE.CONCAT: ops.ConcatPruner(),
            ops.OPTYPE.SPLIT: ops.SplitPruner(),
            ops.OPTYPE.ELEMENTWISE: ops.ElementWisePruner(),
            ops.OPTYPE.RESHAPE: ops.ReshapePruner(),
            ops.OPTYPE.UNBIND: ops.UnbindPruner(),
            ops.OPTYPE.EXPAND: ops.ExpandPruner(),
            ops.OPTYPE.CUSTOMIZED: ops.CustomizedPruner(), # just a placeholder
            ops.OPTYPE.SLICE: ops.SlicePruner(),
            ops.OPTYPE.OUTPUT: ops.OutputPruner(),
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
        
        # For shape inference
        self._2d_4d = True # for pytorch<=1.8

        self.ops = ops 

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
        
        # Ignore layers & nn.Parameter
        if ignored_layers is not None:
            self.IGNORED_LAYERS_IN_TRACING.extend(ignored_layers)
        self.ignored_params = ignored_params if ignored_params is not None else []

        # Ignore all sub-modules of customized layers since they will be handled by the customized pruner
        for layer_type_or_instance in self.CUSTOMIZED_PRUNERS.keys():            
            for m in self.model.modules():
                # check if the module is the target layer or a instance of the layer type
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
        shape_infer.init_shape_information(self)
        # Update index mapping for torch.cat/split/chunck/...
        index_mapping.update_index_mapping(self)
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
        
        # Keep the root indices for index mapping. This will be useful for torch.cat/split/chunck/...
        idxs = [ _helpers._HybridIndex(idx=i, root_idx=i) for i in idxs ] # idxs == root_idxs for the root layer

        # Update index mapping before creating the group
        index_mapping.update_index_mapping(self)
        
        # the root pruning operation
        group = Group()
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
        merged_group = Group() # craft a new group for merging
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

        # create a .root_idxs attribute for each group item to store the root indices
        for i in range(len(merged_group)):
            hybrid_idxs = merged_group[i].idxs
            idxs = _helpers.to_plain_idxs(hybrid_idxs)
            root_idxs = _helpers.to_root_idxs(hybrid_idxs)
            merged_group[i] = _helpers.GroupItem(merged_group[i].dep, idxs) # transform _helpers._HybridIndex to plain index
            merged_group[i].root_idxs = root_idxs
        return merged_group

    def get_all_groups(self, ignored_layers=[], root_module_types=(ops.TORCH_CONV, ops.TORCH_LINEAR)):
        """
            Get all pruning groups for the given module. Groups are generated based on root module types.
            All groups will contain a full indices of the prunable elements or channels.

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
            
            # use output pruning as the root
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
        """Get the pruner for a specific module."""
        p = self.CUSTOMIZED_PRUNERS.get(module.__class__, None) # customized pruners for a specific layer type
        if p is None:
            p = self.REGISTERED_PRUNERS.get(ops.module2type(module), None) # standard pruners
        return p

    def get_out_channels(self, module_or_node):
        """Get the output channels of a module or node."""
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
        """Get the input channels of a module or node."""
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

    def _detect_unwrapped_parameters(self, unwrapped_parameters):
        """Detect unwrapped nn.Parameters that are not part of standard modules."""
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
        """Build the dependency graph."""
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
        """ Trace the model as a graph
        """
        model.eval()
        gradfn2module = {}
        visited = {}
        self._2d_4d = True # for pytorch<=1.8
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

        # Forward the model and record all modules
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
                module2node, o, gradfn2module, reused, visited=visited)

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

    def _trace_computational_graph(self, module2node, output, gradfn2module, reused, visited=set()):
        grad_fn_root = output.grad_fn
        def create_node_if_not_exists(grad_fn):
            module = gradfn2module.get(grad_fn, None)
            if module is not None and module in module2node and module not in reused:
                return module2node[module]

            # 1. link grad_fns and modules
            if module is None:  # a new module
                if grad_fn is grad_fn_root:
                    module = ops._OutputOp(self._op_id, shape=output.shape)
                    self._op_id+=1
                elif not hasattr(grad_fn, "name"):
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
                elif "expand" in grad_fn.name().lower():
                    module = ops._ExpandOp(self._op_id)
                    self._op_id+=1
                elif "view" in grad_fn.name().lower() or 'reshape' in grad_fn.name().lower():
                    module = ops._ReshapeOp(self._op_id)
                    self._op_id+=1
                elif "slice" in grad_fn.name().lower() and "copyslices" not in grad_fn.name().lower():
                    if hasattr(grad_fn, '_saved_start') and hasattr(grad_fn, '_saved_end') and hasattr(grad_fn, '_saved_step') and hasattr(grad_fn, '_saved_dim'):
                        module = ops._SliceOp(self._op_id, grad_fn)
                    else: # for old version of pytorch, we can not handle the slice operation
                        module = ops._ElementWiseOp(self._op_id, grad_fn.name())
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
    

    
