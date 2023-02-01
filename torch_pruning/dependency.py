import torch
import torch.nn as nn
import typing
from enum import IntEnum
from numbers import Number
import warnings

from .pruner import function
from . import _helpers, utils
from . import ops

__all__ = ["Dependency", "DependencyGroup", "DependencyGraph"]


class Node(object):
    """ Node stores the information of a nn.Module like nn.Conv2d or an operations like torch.cat. 
    """

    def __init__(self, module, grad_fn, name=None):
        # Computational Graph
        self.inputs = []
        self.outputs = []
        self.module = module
        self.grad_fn = grad_fn
        self._name = name
        self.type = ops.module2type(module)
        self.class_type = module.__class__

        # Dependency Graph
        self.dependencies = []  # Adjacency List
        self.enable_index_mapping = True

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
        if node not in self.inputs:
            self.inputs.append(node)

    def add_output(self, node):
        if node not in self.outputs:
            self.outputs.append(node)

    def __repr__(self):
        return "<Node: ({})>".format(self.name)

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
        fmt += "\tEnable_index_mapping={}\n".format(
            self.enable_index_mapping)
        fmt = "-" * 32 + "\n"
        return fmt

class Edge():  # for readability
    pass


class Dependency(Edge):
    def __init__(
        self,
        trigger,
        handler,
        source: Node,
        target: Node,
        index_mapping: typing.Callable = None,
    ):
        """Layer dependency (Edge of DepGraph) in structral neural network pruning. 

        Args:
            trigger (Callable or None): the pruning function that triggers this dependency
            handler (Callable): the pruning function that fixes the broken dependency
            source (nn.Module): the source node pruned by trigger
            target (nn.Module): the target node pruned by handler
            index_mapping (Callable): a callable function to transform the pruning indices
        """
        self.trigger = trigger
        self.handler = handler
        self.source = source
        self.target = target
        self.index_mapping = index_mapping

    def __call__(self, idxs: list):
        result = self.handler(
            self.target.module,
            idxs,
        )
        return result

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "[DEP] {} on {} => {} on {}".format(
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

    def __hash__(self):
        return hash((self.source, self.target, self.trigger, self.handler))


class DependencyGroup(object):
    """A group of dependicies with inter-dependency.
    """

    def __init__(self):
        self._group = list()

    def exec(self):
        for dep, idxs in self._group:
            dep(idxs)

    def __call__(self):
        return self.exec()

    def add_dep(self, dep, idxs):
        self._group.append((dep, idxs))

    def __getitem__(self, k):
        return self._group[k]

    @property
    def items(self):
        return self._group

    def has_dep(self, dep):
        for _dep, _ in self._group:
            if dep == _dep:
                return True
        return False

    def has_pruning_op(self, dep, idxs):
        for _dep, _idxs in self._group:
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
                self._group[i] = (_dep, list(set(_idxs + idxs)))
                return
        self.add_dep(dep, idxs)

    def __str__(self):
        fmt = ""
        fmt += "\n" + "-" * 32 + "\n"
        fmt += " " * 10 + "Pruning Group"
        fmt += "\n" + "-" * 32 + "\n"
        for i, (dep, idxs) in enumerate(self._group):
            fmt += "[{}] {}, #Pruned={}\n".format(i, dep, len(idxs))
        fmt += "-" * 32 + "\n"
        return fmt


class DependencyGraph(object):

    def __init__(self):
        _dummy_pruners = {
            ops.OPTYPE.CONCAT: ops.ConcatPruner(),
            ops.OPTYPE.SPLIT: ops.SplitPruner(),
            ops.OPTYPE.ELEMENTWISE: ops.ElementWisePruner(),
            ops.OPTYPE.CUSTOMIZED: None,
        }
        self.REGISTERED_PRUNERS = function.PrunerBox.copy()  # shallow copy
        self.REGISTERED_PRUNERS.update(_dummy_pruners)
        self.CUSTOMIZED_PRUNERS = {}
        self.IGNORED_LAYERS = []

    def build_dependency(
        self,
        model: torch.nn.Module,
        example_inputs: typing.Union[torch.Tensor, typing.Sequence],
        forward_fn: typing.Callable[[
            torch.nn.Module, typing.Union[torch.Tensor, typing.Sequence]], torch.Tensor] = None,
        output_transform: typing.Callable = None,
        unwrapped_parameters: typing.List[nn.Parameter] = None,
        customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None,
        verbose: bool = True,
    ):
        """Build a dependency graph by tracing.

        Args:
            model (class): the model to be pruned.
            example_inputs (torch.Tensor or List): dummy inputs for tracing.
            forward_fn (Callable): a function to run the model with example_inputs, which returns a reduced tensor
            output_transform (Callable): a function to transform network outputs.
            unwrapped_parameters (List): unwrapped parameters defined by users.
            customized_pruners (typing.Dict[typing.Any, function.BasePruningFunc]): pruners for customized layers.
            verbose (Callable): verbose mode.
        """

        self.verbose = verbose
        self.model = model
        self._module2name = {module: name for (
            name, module) in model.named_modules()}
    
        # user-defined nn.Parameters & customized modules
        if unwrapped_parameters is None:
            unwrapped_parameters = []
        self.unwrapped_parameters = unwrapped_parameters
        if customized_pruners is not None:
            for customized_module, customized_pruner in self.customized_pruners.items():
                self.register_customized_layer(customized_module, customized_pruner)
        
        # Ignore sub-modules of customized layers
        for layer_type in self.CUSTOMIZED_PRUNERS.keys():
            for m in self.model.modules():
                if isinstance(m, layer_type):
                    for sub_module in m.modules():
                        if sub_module!=m: self.IGNORED_LAYERS.append(sub_module)

        # Build computational graph by tracing.
        self.module2node = self._trace(
            model, example_inputs, forward_fn, output_transform=output_transform
        )

        # Build the dependency graph
        self._build_dependency(self.module2node)

        # Update index transform
        self.update_index_mapping()
        return self

    def register_customized_layer(
        self,
        layer_type,
        layer_pruner,
    ):
        """Register a customized layer for pruning.

        Args:
            layer_type (class): the type of layer
            pruner (tp.pruner.BasePruningFunc): a pruner for the given layer type.
        """
        self.CUSTOMIZED_PRUNERS[layer_type] = layer_pruner
        
    def check_pruning_group(self, group):
        for dep, idxs in group:
            if function.is_out_channel_pruner(dep.handler):
                prunable_chs = self.get_out_channels(
                    dep.target.module)
                if prunable_chs <= len(idxs):
                    return False
            if function.is_in_channel_pruner(dep.handler):
                prunable_in_chs = self.get_in_channels( 
                    dep.target.module)
                if prunable_in_chs <= len(idxs):
                    return False
        return True

    def is_out_channel_pruner(self, pruner):
        return function.is_out_channel_pruner(pruner)
    
    def is_in_channel_pruner(self, pruner):
        return function.is_in_channel_pruner(pruner)

    def get_pruning_plan(self, module, pruning_fn, idxs):
        return self.get_pruning_group(module, pruning_fn, idxs)

    def get_pruning_group(
        self,
        module: nn.Module,
        pruning_fn: typing.Callable,
        idxs: typing.Union[list, tuple],
    ):
        """Get the pruning group of pruning_fn with the dependency graph.

        Args:
            module (nn.Module): the module to be pruned.
            pruning_fn (Callable): the pruning function.
            idxs (list or tuple): the indices of parameters to be pruned.
        """
        if isinstance(module, ops.TORCH_CONV) and module.groups == module.out_channels:
            pruning_fn = function.prune_depthwise_conv_out_channels
        if isinstance(idxs, Number):
            idxs = [idxs]

        self.update_index_mapping()
        group = DependencyGroup()
        #  the user pruning operation
        root_node = self.module2node[module]
        group.add_dep(
            Dependency(pruning_fn, pruning_fn,
                       source=root_node, target=root_node), idxs
        )

        visited_node = set()

        def _fix_dependency_graph_non_recursive(dep, idxs):
            processing_stack = [(dep, idxs)]
            while len(processing_stack) > 0:
                dep, idxs = processing_stack.pop(-1)
                node, fn = dep.target, dep.handler
                visited_node.add(node)

                for new_dep in node.dependencies:
                    if new_dep.is_triggered_by(fn):
                        new_indices = (
                            new_dep.index_mapping(idxs)
                            if new_dep.index_mapping is not None
                            else idxs
                        )
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
        merged_group = DependencyGroup()
        for dep, idxs in group.items:
            merged_group.add_and_merge(dep, idxs)
        return merged_group

    def get_module_pruner(self, module):
        return self.REGISTERED_PRUNERS.get( ops.module2type(module), None )

    def get_out_channels(self, module_or_node):
        if isinstance(module_or_node, Node):
            module = module_or_node.module
        else:
            module = module_or_node
        p = function.PrunerBox.get(ops.module2type(module), None)
        if p is None:
            return None
        return p.get_out_channels(module)
    
    def get_in_channels(self, module_or_node):
        if isinstance(module_or_node, Node):
            module = module_or_node.module
        else:
            module = module_or_node
        p = function.PrunerBox.get(ops.module2type(module), None)
        if p is None:
            return None
        return p.get_in_channels(module)
    

    def _infer_out_channels_recursively(self, node: Node):
        """ infer the number of output channels recursively
        """
        ch = self.get_out_channels(node)
        if ch is None:
            ch = 0
            for in_node in node.inputs:
                if node.type == ops.OPTYPE.CONCAT:
                    ch += self._infer_out_channels_recursively(in_node)
                else:
                    ch = self._infer_out_channels_recursively(in_node)
            if ch==0:
                return None
        return ch


    def _infer_in_channels_recursively(self, node: Node):
        """ infer the number of input channels recursively
        """
        ch = self.get_in_channels(node)
        if ch is None:
            ch = 0
            for out_node in node.outputs:
                if node.type == ops.OPTYPE.SPLIT:
                    ch += self._infer_in_channels_recursively(out_node)
                else:
                    ch = self._infer_in_channels_recursively(out_node)
            if ch==0:
                return None
        return ch

    def _build_dependency(self, module2node):

        # There will be a dependency between two pruning operations if they:
        # 1) connects to each other in the computational graph or
        # 2) are equivalent, i.e., applied to the same layer and works in the same way.
        # Note that for some units like BN and PReLU, pruning output channels is equivalent to pruning output_channels
        # Rule 2) is designed for this case.

        for _, node in module2node.items():
            # Rule 1) - Input connections
            for in_node in node.inputs:
                handler = self.REGISTERED_PRUNERS.get(in_node.type)
                if handler is None:
                    handler = self.CUSTOMIZED_PRUNERS[in_node.class_type]
                handler = handler.prune_out_channels

                trigger = self.REGISTERED_PRUNERS.get(node.type)
                if trigger is None:
                    trigger = self.CUSTOMIZED_PRUNERS[node.class_type]
                trigger = trigger.prune_in_channels

                dep = Dependency(
                    trigger=trigger, handler=handler, source=node, target=in_node
                )
                node.dependencies.append(dep)

            # Rule 1) - Output connections
            for out_node in node.outputs:
                trigger = self.REGISTERED_PRUNERS.get(node.type)
                if trigger is None:
                    trigger = self.CUSTOMIZED_PRUNERS[node.class_type]
                trigger = trigger.prune_out_channels

                handler = self.REGISTERED_PRUNERS.get(out_node.type)
                if handler is None:
                    handler = self.CUSTOMIZED_PRUNERS[out_node.class_type]
                handler = handler.prune_in_channels

                dep = Dependency(
                    trigger=trigger, handler=handler, source=node, target=out_node
                )
                node.dependencies.append(dep)

            # Rule 2) - out_channel pruner == in_channel pruner
            # if output_channel_pruner and input_channel_pruner are defined to be equivalent，
            # then they should trigger each other.
            # Here we use an identical function for modules with self-dependency,
            # which implicit triggers this rule.

            #p = self.REGISTERED_PRUNERS.get(node.type)
            # if p.prune_out_channels == p.prune_in_channels:
            #    dep = Dependency(
            #        trigger=p.prune_out_channels, handler=p.prune_in_channels, source=node, target=node
            #    )
            #    node.dependencies.append(dep)
            #    dep = Dependency(
            #        trigger=p.prune_in_channels, handler=p.prune_out_channels, source=node, target=node
            #    )
            #    node.dependencies.append(dep)

    def _trace(self, model, example_inputs, forward_fn, output_transform):
        model.eval()
        gradfn2module = {}
        visited = {}

        def _record_grad_fn(module, inputs, outputs):
            if module not in visited:
                visited[module] = 1
            else:
                visited[module] += 1
        
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if isinstance(outputs, torch.nn.utils.rnn.PackedSequence):
                outputs = outputs.data
            gradfn2module[outputs.grad_fn] = module

        registered_types = tuple(ops.type2class(
            t) for t in self.REGISTERED_PRUNERS.keys()) + tuple(self.CUSTOMIZED_PRUNERS.keys())
        hooks = [
            m.register_forward_hook(_record_grad_fn)
            for m in model.modules()
            if (isinstance(m, registered_types) and m not in self.IGNORED_LAYERS)
        ]

        # Feed forward and record gradient functions of prunable modules
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

        # build graph
        if output_transform is not None:
            out = output_transform(out)

        module2node = {}
        for o in utils.flatten_as_list(out):
            self._trace_computational_graph(
                module2node, o.grad_fn, gradfn2module, reused)

        # TODO: Pruning nn.Parameters of Vision transformer
        # This is a corner case for pruning ViT,
        # where concatination of pos_emb and cls_emv is not applied on the feature dim.
        # Notably, this is a bad practice and will be fixed in the future version
        if len(self.unwrapped_parameters) > 0:
            for node in module2node.values():
                if node.type in (ops.OPTYPE.CONCAT, ops.OPTYPE.SPLIT):
                    stack = [node]
                    while len(stack) > 0:
                        n = stack.pop(-1)
                        if n.type == ops.OPTYPE.PARAMETER and len(n.module.shape) == 3:
                            node.enable_index_mapping = False
                            break
                        else:
                            stack.extend(n.inputs)
        return module2node

    def _trace_computational_graph(self, module2node, grad_fn_root, gradfn2module, reused):

        def create_node_if_not_exists(grad_fn):
            module = gradfn2module.get(grad_fn, None)
            if module is not None \
                and module in module2node \
                    and module not in reused:
                return module2node[module]

            # 1. link grad_fns and modules
            if module is None:  # a new module
                if not hasattr(grad_fn, "name"):
                    # we treat unknwon modules as element-wise modules by default,
                    # which does not modify the dimension of features.
                    # If you have some special layers, please register it with DependencyGraph.register_customized_layer
                    module = ops._ElementWiseOp("Unknown")
                    if self.verbose:
                        warnings.warn(
                            "[Warning] Unknown operation {}, treated as an element-wise op".format(
                                str(grad_fn))
                        )
                elif "catbackward" in grad_fn.name().lower():
                    module = ops._ConcatOp()
                elif "split" in grad_fn.name().lower():
                    module = ops._SplitOp()
                else:
                    # treate other ops as element-wise ones, like Add, Sub, Div, Mul.
                    module = ops._ElementWiseOp(grad_fn.name())
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
        visited = set()
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
                            # User-defined nn.Parameter like pos_emb of ViT should be handled carefully
                            is_unwrapped_param = False
                            for (j, p) in enumerate(self.unwrapped_parameters):
                                if f[0].variable is p:
                                    is_unwrapped_param = True
                                    gradfn2module[f[0]] = p
                                    self._module2name[p] = "UserParameter_{}".format(
                                        j)
                            if not is_unwrapped_param:
                                continue
                        input_node = create_node_if_not_exists(f[0])
                        node.add_input(input_node)
                        input_node.add_output(node)
                        processing_stack.append(f[0])
            visited.add(grad_fn)
        return module2node

    def update_index_mapping(self):
        for module, node in self.module2node.items():
            if node.type == ops.OPTYPE.LINEAR:
                # for Conv-Flatten-Linear (e.g., VGG)
                self._update_flatten_index_mapping(node)
            if node.type == ops.OPTYPE.CONCAT:
                self._update_concat_index_mapping(node)
            if node.type == ops.OPTYPE.SPLIT:
                self._update_split_index_mapping(node)

    def _update_flatten_index_mapping(self, fc_node: Node):
        if fc_node.type != ops.OPTYPE.LINEAR:
            return
        fc_in_features = fc_node.module.in_features
        feature_channels = 0
        for n in fc_node.inputs:
            feature_channels = self._infer_out_channels_recursively(n)
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
                        dep.index_mapping = _helpers._FlattenIndexMapping(
                            stride=stride, reverse=True
                        )

                for dep in in_node.dependencies:
                    if dep.target == fc_node:
                        dep.index_mapping = _helpers._FlattenIndexMapping(
                            stride=stride, reverse=False
                        )

    def _update_concat_index_mapping(self, cat_node: Node):
        if cat_node.type != ops.OPTYPE.CONCAT:
            return
        chs = []
        for n in cat_node.inputs:
            chs.append(self._infer_out_channels_recursively(n))
        offsets = [0]
        for ch in chs:
            offsets.append(offsets[-1] + ch)
        cat_node.module.offsets = offsets

        # no transform if the concat dim is different from the feature dim
        for i, in_node in enumerate(cat_node.inputs):
            for dep in cat_node.dependencies:
                if dep.target == in_node:
                    if cat_node.enable_index_mapping:
                        dep.index_mapping = _helpers._ConcatIndexMapping(
                            offset=offsets[i: i + 2], reverse=True
                        )

            for dep in in_node.dependencies:
                if dep.target == cat_node:
                    if cat_node.enable_index_mapping:
                        dep.index_mapping = _helpers._ConcatIndexMapping(
                            offset=offsets[i: i + 2], reverse=False
                        )

    def _update_split_index_mapping(self, split_node: Node):
        if split_node.type != ops.OPTYPE.SPLIT:
            return
        chs = []
        for n in split_node.outputs:
            chs.append(self._infer_in_channels_recursively(n))
        
        offsets = [0]
        for ch in chs:
            offsets.append(offsets[-1] + ch)
        split_node.module.offsets = offsets

        for i, out_node in enumerate(split_node.outputs):
            for dep in split_node.dependencies:
                if dep.target == out_node:
                    if split_node.enable_index_mapping:
                        dep.index_mapping = _helpers._SplitIndexMapping(
                            offset=offsets[i: i + 2], reverse=False
                        )

            for dep in out_node.dependencies:
                if dep.target == split_node:
                    if split_node.enable_index_mapping:
                        dep.index_mapping = _helpers._SplitIndexMapping(
                            offset=offsets[i: i + 2], reverse=True
                        )
