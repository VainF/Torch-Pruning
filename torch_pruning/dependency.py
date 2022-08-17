import torch
import torch.nn as nn
import typing
from enum import IntEnum
from numbers import Number
import warnings

from . import helpers, functional

__all__ = ["PruningClique", "Dependency", "DependencyGraph"]

# Standard Modules
TORCH_CONV = nn.modules.conv._ConvNd
TORCH_BATCHNORM = nn.modules.batchnorm._BatchNorm
TORCH_LAYERNORM = nn.modules.normalization.LayerNorm
TORCH_PRELU = nn.PReLU
TORCH_LINEAR = nn.Linear
TORCH_EMBED = nn.Embedding
try:
    TORCH_MHA = nn.MultiheadAttention
except:
    TORCH_MHA = helpers.DummyMHA  # for pytorch w/o MultiHeadAttention


class OPTYPE(IntEnum):
    CONV = 0
    BN = 1
    LINEAR = 2
    PRELU = 3
    GROUP_CONV = 4

    CONCAT = 5  # torch.cat
    SPLIT = 6  # torch.split
    CUSTOMIZED = 7  # customized module
    ELEMENTWISE = 8  # element-wise add, sub, etc.
    LN = 9  # nn.LayerNorm
    EMBED = 10  # nn.Embedding
    PARAMETER = 11  # nn.Parameter
    MHA = 12


def _module2type(module):
    if isinstance(module, TORCH_CONV):
        if module.groups > 1:
            return OPTYPE.GROUP_CONV
        else:
            return OPTYPE.CONV
    elif isinstance(module, TORCH_BATCHNORM):
        return OPTYPE.BN
    elif isinstance(module, TORCH_PRELU):
        return OPTYPE.PRELU
    elif isinstance(module, TORCH_LINEAR):
        return OPTYPE.LINEAR
    elif isinstance(module, helpers._ConcatOp):
        return OPTYPE.CONCAT
    elif isinstance(module, helpers._SplitOp):
        return OPTYPE.SPLIT
    elif isinstance(module, TORCH_LAYERNORM):
        return OPTYPE.LN
    elif isinstance(module, TORCH_EMBED):
        return OPTYPE.EMBED
    elif isinstance(module, helpers._CustomizedOp):
        return OPTYPE.CUSTOMIZED
    elif isinstance(module, torch.nn.Parameter):
        return OPTYPE.PARAMETER
    elif isinstance(module, TORCH_MHA):
        return OPTYPE.MHA
    else:
        return OPTYPE.ELEMENTWISE


def _infer_out_dim_from_node(node):
    if node.type == OPTYPE.CONV or node.type == OPTYPE.GROUP_CONV:
        return node.module.out_channels
    elif node.type == OPTYPE.BN:
        return node.module.num_features
    elif node.type == OPTYPE.LN:
        return node.module.normalized_shape[functional.prune_layernorm.pruning_dim]
    elif node.type == OPTYPE.LINEAR:
        return node.module.out_features
    elif node.type == OPTYPE.PRELU:
        if node.module.num_parameters == 1:
            return None  # return None if oc can not be infered
        else:
            return node.module.num_parameters
    elif node.type == OPTYPE.PARAMETER:
        return node.module.shape[functional.prune_parameter.dim]
    elif node.type == OPTYPE.CUSTOMIZED:
        return node.customized_pruning_fn["get_out_ch_fn"](node.module)
    elif node.type == OPTYPE.MHA:
        return node.module.embed_dim
    else:
        return None  # return None if oc can not be infered


def _infer_in_dim_from_node(node):
    if node.type == OPTYPE.CONV or node.type == OPTYPE.GROUP_CONV:
        return node.module.in_channels
    elif node.type == OPTYPE.BN:
        return node.module.num_features
    elif node.type == OPTYPE.LN:
        return node.module.normalized_shape[functional.prune_layernorm.pruning_dim]
    elif node.type == OPTYPE.LINEAR:
        return node.module.in_features
    elif node.type == OPTYPE.PRELU:
        if node.module.num_parameters == 1:
            return None  # return None if ic can not be infered
        else:
            return node.module.num_parameters
    elif node.type == OPTYPE.PARAMETER:
        return node.module.shape[functional.prune_parameter.dim]
    elif node.type == OPTYPE.CUSTOMIZED:
        return node.customized_pruning_fn["get_in_ch_fn"](node.module)
    elif node.type == OPTYPE.MHA:
        return node.module.embed_dim
    else:
        return None  # return None if ic can not be infered


######################################################
# Dependency & DependecyGraph
class Node(object):
    def __init__(self, module, grad_fn, name=None):
        self.module = module
        self.grad_fn = grad_fn

        self.inputs = []
        self.outputs = []

        self.dependencies = []

        self._name = name
        self.type = _module2type(module)
        self.enable_index_transform = True

    @property
    def name(self):
        if self._name is None:
            return str(self.module)
        else:
            fmt = self._name
            if self.type != OPTYPE.PARAMETER:
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
        fmt = "<Node: ({})>\n".format(self.name)
        fmt += " " * 4 + "IN:\n"
        for in_node in self.inputs:
            fmt += " " * 8 + "{}\n".format(in_node)
        fmt += " " * 4 + "OUT:\n"
        for out_node in self.outputs:
            fmt += " " * 8 + "{}\n".format(out_node)
        fmt += " " * 4 + "DEP:\n"
        for dep in self.dependencies:
            fmt += " " * 8 + "{}\n".format(dep)
        fmt += "\tEnable_index_transform={}\n".format(self.enable_index_transform)
        return fmt


class Dependency(object):
    def __init__(
        self,
        trigger,
        handler,
        source: Node,
        target: Node,
        index_transform: typing.Callable = None,
    ):
        """Layer dependency in structed neural network pruning.

        Args:
            trigger (Callable or None): a pruning function that breaks the dependency
            handler (Callable): a pruning function to fix the broken dependency
            target (nn.Module): the broken layer
            index_transform (Callable): a function to transform the pruning index
        """
        self.trigger = trigger
        self.handler = handler
        self.source = source
        self.target = target
        self.index_transform = index_transform

    def __call__(self, idxs: list, dry_run: bool = False):
        result = self.handler(
            self.target.module,
            idxs,
            dry_run=dry_run,
        )
        return result

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "[DEP] {} on {} => {} on {}".format(
            "None" if self.trigger is None else self.trigger.__class__.__name__,
            self.source.name,
            self.handler.__class__.__name__,
            self.target.name,
        )

    def is_triggered_by(self, pruning_fn):
        return pruning_fn == self.trigger

    def __eq__(self, other):
        return (
            (self.trigger == other.trigger)
            and self.handler == other.handler
            and self.target == other.target
            # and self.source == other.source
        )


class PruningClique(object):
    """Pruning clique.

    Args:
        dry_run (Callable or None): only return the info about pruning.
        module_to_name (dict): mapping nn.module to a readable name. It will be filled by DependencyGraph.
    """

    def __init__(self):
        self._cliques = list()
        self._metrics_scalar_sum = helpers.ScalarSum()
        self._metrics_vector_sum = helpers.VectorSum()

    def add_clique(self, dep, idxs):
        self._cliques.append((dep, idxs))

    def __getitem__(self, k):
        return self._cliques[k]

    @property
    def items(self):
        return self._cliques

    def exec(self, dry_run=False):
        per_layer_metrics = []
        for dep, idxs in self._cliques:
            _, metric_dict = dep(idxs, dry_run=dry_run)
            per_layer_metrics.append(metric_dict)
        return per_layer_metrics

    def has_dep(self, dep):
        for _dep, _ in self._cliques:
            if dep == _dep:
                return True
        return False

    def has_pruning_op(self, dep, idxs):
        for _dep, _idxs in self._cliques:
            if (
                _dep.target == dep.target
                and _dep.handler == dep.handler
                and _idxs == idxs
            ):
                return True
        return False

    def __len__(self):
        return len(self._cliques)

    def add_and_merge(self, dep, idxs):
        for i, (_dep, _idxs) in enumerate(self._cliques):
            if _dep.target == dep.target and _dep.handler == dep.handler:
                self._cliques[i] = (_dep, list(set(_idxs + idxs)))
                return
        self.add_clique(dep, idxs)

    def __str__(self):
        fmt = ""
        fmt += "\n" + "-" * 32 + "\n"
        fmt += " " * 10 + "Pruning Clique"
        fmt += "\n" + "-" * 32 + "\n"
        self._metrics_scalar_sum.reset()
        self._metrics_vector_sum.reset()

        for i, (dep, idxs) in enumerate(self._cliques):
            _, metric_dict = dep(idxs, dry_run=True)
            for k, v in metric_dict.items():
                if helpers.is_scalar(v):
                    self._metrics_scalar_sum.update(k, v)
                else:
                    self._metrics_vector_sum.update(k, v)
            if i == 0:
                fmt += "User Operation:\n"
            fmt += "[ {}, Index={}, metric={}]\n".format(dep, idxs, metric_dict)
            if i == 0:
                fmt += "\nCoupled pruning:\n"

        scalar_metric = self._metrics_scalar_sum.results()
        vector_metric = self._metrics_vector_sum.results()
        scalar_metric.update(vector_metric)
        fmt += "\nMetric Sum: {}\n".format(scalar_metric)
        fmt += "-" * 32 + "\n"
        return fmt


class DependencyGraph(object):

    # can be updated by users
    PRUNABLE_MODULES = [
        TORCH_CONV,
        TORCH_BATCHNORM,
        TORCH_LINEAR,
        TORCH_PRELU,
        TORCH_LAYERNORM,
        TORCH_EMBED,
        TORCH_MHA,
    ]

    PRUNING_FN = (
        {  # functions that prune (1. input channels,           2. output channels)
            OPTYPE.CONV: (
                functional.prune_conv_in_channel,
                functional.prune_conv_out_channel,
            ),
            OPTYPE.BN: (functional.prune_batchnorm, functional.prune_batchnorm),
            OPTYPE.PRELU: (functional.prune_prelu, functional.prune_prelu),
            OPTYPE.LINEAR: (
                functional.prune_linear_in_channel,
                functional.prune_linear_out_channel,
            ),
            OPTYPE.GROUP_CONV: (
                functional.prune_group_conv,
                functional.prune_group_conv,
            ),
            OPTYPE.CONCAT: (helpers._prune_concat, helpers._prune_concat),
            OPTYPE.SPLIT: (helpers._prune_split, helpers._prune_split),
            OPTYPE.ELEMENTWISE: (
                helpers._prune_elementwise_op,
                helpers._prune_elementwise_op,
            ),
            OPTYPE.LN: (functional.prune_layernorm, functional.prune_layernorm),
            OPTYPE.EMBED: (functional.prune_embedding, functional.prune_embedding),
            OPTYPE.PARAMETER: (functional.prune_parameter, functional.prune_parameter),
            OPTYPE.MHA: (
                functional.prune_multihead_attention,
                functional.prune_multihead_attention,
            ),
            OPTYPE.CUSTOMIZED: (None, None),  # placeholder
        }
    )

    RULES_FOR_SUCCEEDING_LAYERS = {}
    RULES_FOR_PRECEDING_LAYERS = {}
    for t1 in PRUNING_FN.keys():
        for t2 in PRUNING_FN.keys():
            RULES_FOR_SUCCEEDING_LAYERS[(t1, t2)] = (
                PRUNING_FN[t1][1],  # trigger
                PRUNING_FN[t2][0],  # handler
            )  # change in_channels of succeeding layers
            RULES_FOR_PRECEDING_LAYERS[(t1, t2)] = (
                PRUNING_FN[t1][0],  # trigger
                PRUNING_FN[t2][1],  # handler
            )  # change out_channels of preceding layers
    CUSTOMIZED_PRUNING_FN = {}

    @property
    def out_channel_pruners(self):
        return [pruners[1] for pruners in self.PRUNING_FN.values() if pruners[1] is not None]
    
    @property
    def in_channel_pruners(self):
        return [pruners[0] for pruners in self.PRUNING_FN.values() if pruners[0] is not None]

    def build_dependency(
        self,
        model: torch.nn.Module,
        example_inputs: typing.Union[torch.Tensor, typing.Sequence],
        output_transform: typing.Callable = None,
        verbose: bool = True,
        user_defined_parameters=None,
    ):
        """Build a dependency graph by tracing.

        Args:
            model (class): the model to be pruned.
            example_inputs (torch.Tensor or List): dummy inputs for tracing.
            output_transform (Callable): A function to transform network outputs.
            verbose (Callable): verbose mode.
        """

        self.verbose = verbose

        self._module2name = {module: name for (name, module) in model.named_modules()}
        # user-defined nn.Parameters like the learnable pos_emb in ViT
        if user_defined_parameters is None:
            user_defined_parameters = []
        self.user_defined_parameters = user_defined_parameters

        # build dependency graph by tracing
        self.module2node = self._trace(
            model, example_inputs, output_transform=output_transform
        )
        self._build_dependency(self.module2node)
        self.update_index()
        return self

    def register_customized_layer(
        self,
        layer_type,
        in_ch_pruning_fn,
        out_ch_pruning_fn,
        get_in_ch_fn,
        get_out_ch_fn,
    ):
        """Register a customized layer for pruning.

        Args:
            layer_type (class): the type of layer
            in_ch_pruning_fn (Callable): A function to prune channels/dimensions of input tensor
            out_ch_pruning_fn (Callable): A function to prune channels/dimensions of output tensor
            get_in_ch_fn (Callable): estimate the n_channel of layer input. Return None if the layer does not change tensor shape.
            get_out_ch_fn (Callable):estimate the n_channel of layer output. Return None if the layer does not change tensor shape.
        """
        self.CUSTOMIZED_PRUNING_FN[layer_type] = {
            "in_ch_pruning_fn": in_ch_pruning_fn,
            "out_ch_pruning_fn": out_ch_pruning_fn,
            "get_in_ch_fn": get_in_ch_fn,
            "get_out_ch_fn": get_out_ch_fn,
        }
        self.PRUNABLE_MODULES.append(layer_type)

    def check_pruning_clique(self, clique):
        for dep, idxs in clique.items:
            if dep.handler in (
                functional.prune_conv_out_channel,
                functional.prune_batchnorm,
                functional.prune_linear_out_channel,
                functional.prune_group_conv,
            ):
                prunable_chs = count_prunable_out_channels(dep.target.module)
                if prunable_chs <= len(idxs):
                    return False
            if dep.handler in (
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ):
                prunable_in_chs = count_prunable_in_channels(dep.target.module)
                if prunable_in_chs <= len(idxs):
                    return False
        return True

    def get_pruning_clique(self, module, pruning_fn, idxs):
        warnings.warn("get_pruning_clique has been deprecated. Please use get_pruning_clique.")
        return self.get_pruning_clique(module, pruning_fn, idxs)

    def get_pruning_clique(
        self,
        module: nn.Module,
        pruning_fn: typing.Callable,
        idxs: typing.Union[list, tuple],
    ):
        """Get a pruning clique from the dependency graph, according to user's pruning operations.

        Args:
            module (nn.Module): the module to be pruned.
            pruning_fn (Callable): the pruning function.
            idxs (list or tuple): the indices of paramters to be pruned.
        """
        if isinstance(module, TORCH_CONV) and module.groups > 1:
            pruning_fn = functional.prune_group_conv
        if isinstance(idxs, Number):
            idxs = [idxs]

        self.update_index()
        clique = PruningClique()
        #  the user pruning operation
        root_node = self.module2node[module]
        clique.add_clique(
            Dependency(pruning_fn, pruning_fn, source=root_node, target=root_node), idxs
        )

        visited = set()

        def _fix_dependency_graph_non_recursive(node, fn, indices):
            processing_stack = [(node, fn, indices)]
            while len(processing_stack) > 0:
                node, fn, indices = processing_stack.pop(-1)
                # print(node in visited)
                visited.add(node)

                for dep in node.dependencies:
                    if dep.is_triggered_by(fn):
                        new_indices = (
                            dep.index_transform(indices)
                            if dep.index_transform is not None
                            else indices
                        )
                        if len(new_indices) == 0:
                            continue
                        if dep.target in visited and clique.has_pruning_op(
                            dep, new_indices
                        ):
                            continue
                        else:
                            clique.add_clique(dep, new_indices)
                            processing_stack.append(
                                (dep.target, dep.handler, new_indices)
                            )

        _fix_dependency_graph_non_recursive(root_node, pruning_fn, idxs)

        # merge pruning ops
        merged_clique = PruningClique()
        for dep, idxs in clique.items:
            merged_clique.add_and_merge(dep, idxs)
        return merged_clique

    def _build_dependency(self, module2node):
        for module, node in module2node.items():
            for in_node in node.inputs:
                preceding_rule = self.RULES_FOR_PRECEDING_LAYERS.get(
                    (node.type, in_node.type), None
                )
                if preceding_rule is not None:
                    trigger = preceding_rule[0]
                    handler = preceding_rule[1]
                    if trigger is None:
                        trigger = self.CUSTOMIZED_PRUNING_FN[type(node.module)][
                            "in_ch_pruning_fn"
                        ]
                    if handler is None:
                        handler = self.CUSTOMIZED_PRUNING_FN[type(in_node.module)][
                            "out_ch_pruning_fn"
                        ]
                    dep = Dependency(
                        trigger=trigger, handler=handler, source=node, target=in_node
                    )
                    node.dependencies.append(dep)

            for out_node in node.outputs:
                succeeding_rule = self.RULES_FOR_SUCCEEDING_LAYERS.get(
                    (node.type, out_node.type), None
                )
                if succeeding_rule is not None:
                    trigger = succeeding_rule[0]
                    handler = succeeding_rule[1]
                    if trigger is None:
                        trigger = self.CUSTOMIZED_PRUNING_FN[type(node.module)][
                            "out_ch_pruning_fn"
                        ]
                    if handler is None:
                        handler = self.CUSTOMIZED_PRUNING_FN[type(out_node.module)][
                            "in_ch_pruning_fn"
                        ]
                    dep = Dependency(
                        trigger=trigger, handler=handler, source=node, target=out_node
                    )
                    node.dependencies.append(dep)

    def _trace(self, model, example_inputs, output_transform):
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
            gradfn2module[outputs.grad_fn] = module

        hooks = [
            m.register_forward_hook(_record_grad_fn)
            for m in model.modules()
            if isinstance(m, tuple(self.PRUNABLE_MODULES))
        ]

        # Feed forward and record gradient functions of prunable modules
        if isinstance(example_inputs, (tuple, list)):
            out = model(*example_inputs)
        elif isinstance(example_inputs, dict):
            out = model(**example_inputs)
        elif isinstance(example_inputs, torch.Tensor):
            out = model(example_inputs)
        for hook in hooks:
            hook.remove()
        # for recursive models or layers
        reused = [m for (m, count) in visited.items() if count > 1]

        # build graph
        if output_transform is not None:
            out = output_transform(out)
        
        module2node = {}
        for o in flatten_as_list(out):
            self._build_graph(module2node, o.grad_fn, gradfn2module, reused)

        # BUG: Special case for torch.cat in ViT,
        # where concatination is not applied to feature dims.
        # Notably, this is a bad practice and will be fixed in the future version
        # Some problems may occurs if your vision transform has a lot of complicated torch.cat.
        if len(self.user_defined_parameters) > 0:
            for node in module2node.values():
                if node.type in (OPTYPE.CONCAT, OPTYPE.SPLIT):
                    stack = [node]
                    while len(stack) > 0:
                        n = stack.pop(-1)
                        if n.type == OPTYPE.PARAMETER and len(n.module.shape) == 3:
                            node.enable_index_transform = False
                            break
                        else:
                            stack.extend(n.inputs)
        return module2node

    def _build_graph(self, module2node, grad_fn_root, gradfn2module, reused):

        def create_node_if_not_exists(grad_fn):
            module = gradfn2module.get(grad_fn, None)
            if module is not None and module in module2node and module not in reused:
                return module2node[module]

            if module is None:  # unseen modules
                if not hasattr(grad_fn, "name"):
                    # we treat unknwon modules as element-wise modules
                    module = helpers._ElementWiseOp("Unknown")
                    if self.verbose:
                        warnings.warn(
                            "[Warning] Unrecognized operation {} will be treated as an element-wise op".format(str(grad_fn))
                        )
                elif "catbackward" in grad_fn.name().lower():  # concat op
                    module = helpers._ConcatOp()
                elif "split" in grad_fn.name().lower():
                    module = helpers._SplitOp()
                else:
                    # treate other ops as element-wise ones
                    module = helpers._ElementWiseOp(grad_fn.name())
                gradfn2module[grad_fn] = module

            if module not in module2node:  # create nodes
                node = Node(
                    module=module,
                    grad_fn=grad_fn,
                    name=self._module2name.get(module, None),
                )
                if (
                    type(module) in self.CUSTOMIZED_PRUNING_FN.keys()
                ):  # mark it as a customized OP
                    node.type = OPTYPE.CUSTOMIZED
                    node.customized_pruning_fn = self.CUSTOMIZED_PRUNING_FN[
                        type(module)
                    ]
                module2node[module] = node
            else:
                node = module2node[module]
            return node

        # non-recursive graph construction
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
                        ):
                            is_user_defined_param = False
                            for (j, p) in enumerate(self.user_defined_parameters):
                                if f[0].variable is p:
                                    is_user_defined_param = True
                                    gradfn2module[f[0]] = p
                                    self._module2name[p] = "UserParameter_{}".format(j)
                            if not is_user_defined_param:
                                continue
                        input_node = create_node_if_not_exists(f[0])
                        node.add_input(input_node)
                        input_node.add_output(node)
                        processing_stack.append(f[0])
            visited.add(grad_fn)
        return module2node

    def update_index(self):
        for module, node in self.module2node.items():
            if node.type == OPTYPE.LINEAR:
                self._set_fc_index_transform(node)
            if node.type == OPTYPE.CONCAT:
                self._set_concat_index_transform(node)
            if node.type == OPTYPE.SPLIT:
                self._set_split_index_transform(node)

    def _set_fc_index_transform(self, fc_node: Node):
        if fc_node.type != OPTYPE.LINEAR:
            return
        fc_in_features = fc_node.module.in_features
        feature_channels = 0
        for n in fc_node.inputs: 
            feature_channels = _infer_out_dim_from_node_by_recursion(n)
            if feature_channels>0: # =0 if there is a residual connection to model inputs
                break 
        if (
            feature_channels <= 0
        ):  # the first layer: https://github.com/VainF/Torch-Pruning/issues/21
            return
        stride = fc_in_features // feature_channels
        if stride > 1 and fc_in_features % feature_channels==0:
            for in_node in fc_node.inputs:
                for dep in fc_node.dependencies:
                    if dep.target == in_node:
                        dep.index_transform = helpers._FlattenIndexTransform(
                            stride=stride, reverse=True
                        )

                for dep in in_node.dependencies:
                    if dep.target == fc_node:
                        dep.index_transform = helpers._FlattenIndexTransform(
                            stride=stride, reverse=False
                        )

    def _set_concat_index_transform(self, cat_node: Node):
        if cat_node.type != OPTYPE.CONCAT:
            return
        chs = []
        for n in cat_node.inputs:
            chs.append(_infer_out_dim_from_node_by_recursion(n))
        offsets = [0]
        for ch in chs:
            offsets.append(offsets[-1] + ch)
        cat_node.module.offsets = offsets

        # no transform if the concat dim is different from the feature dim
        for i, in_node in enumerate(cat_node.inputs):
            for dep in cat_node.dependencies:
                if dep.target == in_node:
                    if cat_node.enable_index_transform:
                        dep.index_transform = helpers._ConcatIndexTransform(
                            offset=offsets[i : i + 2], reverse=True
                        )

            for dep in in_node.dependencies:
                if dep.target == cat_node:
                    if cat_node.enable_index_transform:
                        dep.index_transform = helpers._ConcatIndexTransform(
                            offset=offsets[i : i + 2], reverse=False
                        )

    def _set_split_index_transform(self, split_node: Node):
        if split_node.type != OPTYPE.SPLIT:
            return

        chs = []
        for n in split_node.outputs:
            chs.append(_infer_in_dim_from_node_by_recursion(n))

        offsets = [0]
        for ch in chs:
            offsets.append(offsets[-1] + ch)
        split_node.module.offsets = offsets

        for i, out_node in enumerate(split_node.outputs):
            for dep in split_node.dependencies:
                if dep.target == out_node:
                    if split_node.enable_index_transform:
                        dep.index_transform = helpers._SplitIndexTransform(
                            offset=offsets[i : i + 2], reverse=False
                        )

            for dep in out_node.dependencies:
                if dep.target == split_node:
                    if split_node.enable_index_transform:
                        dep.index_transform = helpers._SplitIndexTransform(
                            offset=offsets[i : i + 2], reverse=True
                        )


def _infer_out_dim_from_node_by_recursion(node):
    ch = _infer_out_dim_from_node(node)
    if ch is None:
        ch = 0
        for in_node in node.inputs:
            if node.type == OPTYPE.CONCAT:
                ch += _infer_out_dim_from_node_by_recursion(in_node)
            else:
                ch = _infer_out_dim_from_node_by_recursion(in_node)
    return ch


def _infer_in_dim_from_node_by_recursion(node):
    ch = _infer_in_dim_from_node(node)
    if ch is None:
        ch = 0
        for out_node in node.outputs:
            if node.type == OPTYPE.SPLIT:
                ch += _infer_in_dim_from_node_by_recursion(out_node)
            else:
                ch = _infer_in_dim_from_node_by_recursion(out_node)
    return ch


def flatten_as_list(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    elif isinstance(obj, (list, tuple)):
        flattened_list = []
        for sub_obj in obj:
            flattened_list.extend(flatten_as_list(sub_obj))
        return flattened_list
    elif isinstance(obj, dict):
        flattened_list = []
        for sub_obj in obj.values():
            flattened_list.extend(flatten_as_list(sub_obj))
        return flattened_list
    else:
        return obj


def count_prunable_out_channels(module):
    if isinstance(module, TORCH_CONV):
        return module.weight.shape[0]
    elif isinstance(module, TORCH_LINEAR):
        return module.out_features
    elif isinstance(module, TORCH_BATCHNORM):
        return module.num_features
    elif isinstance(module, TORCH_PRELU):
        if len(module.weight) == 1:
            return 0
        else:
            return len(module.weight)
    else:
        return 0


def count_prunable_in_channels(module):
    if isinstance(module, TORCH_CONV):
        return module.weight.shape[1]
    elif isinstance(module, TORCH_LINEAR):
        return module.in_features
    elif isinstance(module, TORCH_BATCHNORM):
        return module.num_features
    elif isinstance(module, TORCH_PRELU):
        if len(module.weight) == 1:
            return 0
        else:
            return len(module.weight)
    else:
        return 0
