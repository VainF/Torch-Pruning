import torch
import torch.nn as nn
from typing import Callable
from functools import reduce
from operator import mul

from .prune_fn import *

__all__ = ['PruningPlan', 'Dependency', 'DependencyGraph']

TORCH_CONV = nn.modules.conv._ConvNd
TORCH_BATCHNORM = nn.modules.batchnorm._BatchNorm
TORCH_PRELU = nn.PReLU
TORCH_LINEAR = nn.Linear

class PruningPlan(list):
    """ Pruning plan.
    
    Args:
        dry_run (Callable or None): only return the info about pruning.
        module_to_name (dict): mapping nn.module to a readable name. It will be filled by DependencyGraph.
    """

    def __init__(self, dry_run=False, module_to_name=None):
        super(PruningPlan, self).__init__()
        self._module_to_name = module_to_name

    def append(self, x):
        assert len(x) == 2 and isinstance(x[0], Dependency) and isinstance(
            x[1], (list, tuple)), "a pruning plan must consists of a dependency and a index list"
        if x not in self:
            super(PruningPlan, self).append(x)
        else:
            print("Warning: ignore the existed plan: (%s, %s)" % x)

    def exec(self, dry_run=False):
        return self(dry_run)

    def __call__(self, dry_run=False):
        num_pruned = 0
        for dep, idxs in self:
            _, n = dep(idxs, dry_run=dry_run)
            num_pruned += n
        return num_pruned

    def _get_module_name(self, module):
        if isinstance(module, _ElementWiseOp):
            return 'elementwise'
        elif isinstance(module, _ConcatOp):
            return 'concat'
        elif self._module_to_name is not None:
            return self._module_to_name[module]
        else:
            return str(module)

    def __str__(self):
        str_result = ""
        str_result += "\n-------------\n"
        totally_pruned = 0
        for dep, idxs in self:
            _, n_pruned = dep(idxs, dry_run=True)
            totally_pruned += n_pruned
            if dep.index_mapping is not None:
                idxs = dep.index_mapping(idxs)
            str_result += "[ %s on %s (%s), Index=%s, NumPruned=%d]\n" % (
                dep.handler.__name__, self._get_module_name(dep.target), dep.target, idxs, n_pruned)

        str_result += "%d parameters will be pruned\n" % (totally_pruned)
        str_result += "-------------\n"
        return str_result


class Dependency(object):
    def __init__(self, condition, handler, target):
        """ Layer dependency in structed neural network pruning. 
        Pruning on some layers may break dependencies. This class 
        provides the required operations to fix dependencies.

        Parameters:

            condition (Callable or None): a pruning function which will break this dependency 
            handler (Callable): a pruning function to fix the dependency
            target (nn.Module): the target layer
        """
        self.condition = condition
        self.handler = handler
        self.target = target

        self.index_mapping = None
        self.offset = 0

    def __call__(self, idxs, dry_run=False):
        if self.index_mapping is not None:
            idxs = self.index_mapping(idxs)
        ret = self.handler(self.target, idxs, dry_run=dry_run)
        # update offset after pruning
        if not dry_run and isinstance(self.target, _ConcatOp):
            for i in range(len(self.target.offset)):
                for idx in idxs:
                    if self.target.offset[i] > idx:
                        self.target.offset[i] -= 1
        return ret

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "%s => %s on %s" % ("None" if self.condition is None else self.condition.__name__, self.handler.__name__, self.target)

    def check(self, pruning_fn):
        return pruning_fn == self.condition

    def __eq__(self, other):
        return (((self.condition is None) or self.condition == other.condition) and self.handler == other.handler and self.target == other.target)


def _prune_concat(layer, *args, **kargs):
    return layer, 0


def _prune_elementwise_op(layer, *args, **kargs):
    return layer, 0

# dummy module
class _ConcatOp(nn.Module):
    def __init__(self):
        super(_ConcatOp, self).__init__()
        self.offset = None

    def __repr__(self):
        return "ConcatOp(%s)" % self.offset

class _ElementWiseOp(nn.Module):
    def __init__(self):
        super(_ElementWiseOp, self).__init__()


class _FalttenIndexMapping(object):
    def __init__(self, stride):
        self.stride = stride

    def __call__(self, idxs):
        new_idxs = []
        for idx in idxs:
            new_idxs.extend(list(range(idx*self.stride, (idx+1)*self.stride)))
        return new_idxs


class DependencyGraph(object):
    r""" Layer dependency graph for pytorch model

    Args:
        model (nn.Module): a pytorch model
        fake_input (torch.Tensor): a fake input for the pytorch model
        ignore_conv_fc_dep (bool): ignore the dependency between convolutional layer and fully-connected layer.
    """
    PRUNABLE_MODULES = (TORCH_CONV, TORCH_BATCHNORM, TORCH_PRELU, TORCH_LINEAR,)
    PENDING_GRAD_FN_TYPE = ('ConvolutionBackward', 'BatchNormBackward', 'PreluBackward', 'MmBackward')
    PENDING_ELEMENTWISE_GRAD_FN_TYPE = ('AddBackward', 'SubBackward', 'MulBackward', 'DivBackward')
    PENDING_CONCAT_GRAD_FN_TYPE = ('CatBackward',)

    ALL_PENDING_GRAD_FN_TYPE = PENDING_GRAD_FN_TYPE + \
        PENDING_ELEMENTWISE_GRAD_FN_TYPE + PENDING_CONCAT_GRAD_FN_TYPE

    # Rules for Pruning module 1
    #                                   ( change input shape,  change output shape )
    HANDLER = {'ConvolutionBackward':   (prune_related_conv,   prune_conv),
               'BatchNormBackward':     (prune_batchnorm,      prune_batchnorm),
               'PreluBackward':         (prune_prelu,          prune_prelu),
               'ElementWiseOpBackward': (_prune_elementwise_op, _prune_elementwise_op),
               'MmBackward':            (prune_related_linear, prune_linear),
               'CatBackward':           (_prune_concat, _prune_concat),
               }
    RULES = {}
    REVERSE_RULES = {}

    for k1 in HANDLER.keys():
        for k2 in HANDLER.keys():
            RULES[k1+k2] = (HANDLER[k1][1], HANDLER[k2][0])
            if k2 != 'CatBackward':
                REVERSE_RULES[k1+k2] = (HANDLER[k2][0], HANDLER[k1][1])

    def __init__(self, model, fake_input, ignore_conv_fc_dep=False):
        self.dependencies = {}
        self.ignore_conv_fc_dep = ignore_conv_fc_dep
        self.build_dependency(model, fake_input)

    def get_pruning_plan(self, module, pruning_fn, idxs):
        """get a pruning plan for the user's pruning operation. 

        Args:
            module (nn.Module): the module to be pruned.
            pruning_fn (callable): the pruning function.
            idxs (list): pruning index.
        """

        visited_module = set()
        triggered_deps = PruningPlan(module_to_name=self.module_to_name)
        # manually specified pruning operation
        triggered_deps.append((Dependency(None, pruning_fn, module), idxs))

        def _iterate_denpendency_tree(m, pf, i):
            visited_module.add(m)
            deps = self.dependencies.get(m, [])
            for dep in deps:
                if dep.check(pf) and dep.target not in visited_module:
                    new_i = [ii+dep.offset for ii in i]
                    triggered_deps.append((dep, new_i))
                    _iterate_denpendency_tree(dep.target, dep.handler, new_i)
        _iterate_denpendency_tree(module, pruning_fn, idxs)
        return triggered_deps

    def _get_module_name(self, module):
        if isinstance(module, _ElementWiseOp):
            return 'elementwise'
        elif isinstance(module, _ConcatOp):
            return 'concat'
        else:
            return self.module_to_name.get(module, None)

    def print_dependency(self, source=None, verbose=True):
        """print layer dependencies 

        Args:
            source (nn.Module): the source module. print all dependencies if source is None.
            verbose (bool): whether to print detailed model information
        """
        def _print(source, deps):
            if verbose:
                print("\n-------------")
                print("%s (%s):" % (self._get_module_name(source), source))
                for dep in deps:
                    print("[ %s => %s on %s (%s) ]" % (dep.condition.__name__, dep.handler.__name__, self._get_module_name(dep.target), dep.target))
                print("-------------")
            else:
                print("\n-------------")
                print("%s:" % self._get_module_name(source))
                for dep in deps:
                    print("[ %s => %s on %s ]" % (dep.condition.__name__, dep.handler.__name__, self._get_module_name(dep.target)))
                print("-------------")

        if source is not None:
            deps = self.dependencies[source]
            _print(source, deps)
            return

        for source, deps in self.dependencies.items():
            _print(source, deps)

    def add_dependency(self, source_module, target_module, dep):
        """Add a dependency to the target module. It can used to customize dependency graph.

        Args:
            source_module (nn.Module): source of the dependency
            target_module (nn.Module): target of the dependency
            dep (pruning.Dependency): the dependency object
        """

        dep.target = target_module
        deps = self.dependencies.get(source_module, None)
        if deps:
            if dep not in deps:
                deps.append(dep)
        else:
            self.dependencies[source_module] = [dep, ]

    def _get_dependency(self, source_fn_name, target_fn_name, reverse=False):
        if not reverse:
            dep_prune_fn = self.RULES.get(source_fn_name+target_fn_name)
        else:
            dep_prune_fn = self.REVERSE_RULES.get(source_fn_name+target_fn_name)

        if dep_prune_fn:
            return Dependency(*dep_prune_fn, target=None)
        return dep_prune_fn

    def build_dependency(self, model, fake_input):
        """build pruning dependencies for the model. A fake input is required to obatin the dynamic graph.

        Args:
            model (nn.Module): a pytorch model
            fake_input (torch.Tensor): fake input 
        """

        self.model = model
        self.dependencies = {}

        grad_fn_to_module = {}
        extractor_feature_shape = None
        fc_after_conv = False
        self.conv_fc_stride = {}

        def record_grad_fn(module, input, output):
            nonlocal extractor_feature_shape, fc_after_conv
            if isinstance(module, self.PRUNABLE_MODULES):
                grad_fn_to_module[output.grad_fn] = module
            
            if len(output.shape) > 2:
                extractor_feature_shape = output.shape
                fc_after_conv = True

            if isinstance(module, TORCH_LINEAR) and fc_after_conv == True:
                fc_after_conv = False
                # reshape
                if reduce(mul, extractor_feature_shape[1:])==input[0].shape[1]:
                    self.conv_fc_stride[module] = reduce(mul, extractor_feature_shape[2:])
                # global pooling
                elif extractor_feature_shape[1] == input[0].shape[1]:
                    self.conv_fc_stride[module] = 1
                else:
                    self.conv_fc_stride[module] = -1
                    print("Warning: Unrecognized Conv-FC Dependency. Please handle the dependency manually")
        hooks = []
        for m in model.modules():
            hooks.append(m.register_forward_hook(record_grad_fn))
        out = model(fake_input)
        for hook in hooks:
            hook.remove()

        self.grad_fn_to_module = grad_fn_to_module
        self._traverse_graph(out.grad_fn)

        # model to key for print
        self.module_to_name = {}
        for name, module in self.model.named_modules():
            self.module_to_name[module] = name

    def _get_grad_fn_name(self, grad_fn):
        """map grad_fn to a string
        """
        grad_fn_name = grad_fn.name().lower()
        for FN_NAME in self.ALL_PENDING_GRAD_FN_TYPE:
            if FN_NAME.lower() in grad_fn_name:
                if FN_NAME in self.PENDING_ELEMENTWISE_GRAD_FN_TYPE:
                    return 'ElementWiseOpBackward'
                return FN_NAME
        return None

    def _traverse_graph(self, begin_nodes):
        """traverse the grad_fn graph and build layer depenpencies
        """
        # collect nodes
        visited = set()

        def _calc_num_channels_and_index_offset(n, previous_ch):
            module = self.grad_fn_to_module.get(n, None)
            if module is None or isinstance(module, (TORCH_BATCHNORM, TORCH_PRELU)):
                # print(previous_ch)
                if len(previous_ch) > 0:
                    return sum(previous_ch)
                else:
                    return 0
            elif isinstance(module, TORCH_LINEAR):
                return module.out_features
            elif isinstance(module, TORCH_CONV):
                return module.out_channels
            elif isinstance(module, _ConcatOp):
                module.offset = [0]
                module.offset.extend(previous_ch)
                for i in range(1, len(module.offset)):
                    module.offset[i] += module.offset[i-1]
                return sum(previous_ch)

        def _collect_pending_nodes_and_model_info(node):
            if node not in visited:
                visited.add(node)
            else:
                return grad_fn_to_channels[node]
            # print(node)
            grad_fn_name = self._get_grad_fn_name(node)

            if grad_fn_name == 'ElementWiseOpBackward' and node not in self.grad_fn_to_module.keys():
                # create a Dummy Module
                self.grad_fn_to_module[node] = _ElementWiseOp()

            elif grad_fn_name == 'CatBackward' and node not in self.grad_fn_to_module.keys():
                self.grad_fn_to_module[node] = _ConcatOp()

            if grad_fn_name is not None:
                pending_nodes.add(node)

            previous_channels = []
            if hasattr(node, 'next_functions'):
                for u in node.next_functions:
                    if u[0] is not None:
                        ch = _collect_pending_nodes_and_model_info(u[0])
                        if ch is None:
                            ch = 0
                        previous_channels.append(ch)

            cur_channels = _calc_num_channels_and_index_offset(node, previous_channels)

            grad_fn_to_channels[node] = cur_channels
            return cur_channels

        pending_nodes = set()
        grad_fn_to_channels = {}
        _collect_pending_nodes_and_model_info(begin_nodes)

        def _recursively_detect_dependencies(node, path_id):
            if begin_node != node:
                begin_node_name = self._get_grad_fn_name(begin_node)
                node_name = self._get_grad_fn_name(node)
                if node_name and begin_node_name:
                    rev_dep = self._get_dependency(node_name, begin_node_name, reverse=True)
                    if rev_dep is not None:
                        node_module = self.grad_fn_to_module[node]
                        begin_node_module = self.grad_fn_to_module[begin_node]
                        self.add_dependency(begin_node_module, node_module, rev_dep)

                    dep = self._get_dependency(node_name, begin_node_name)
                    if dep is not None:
                        node_module = self.grad_fn_to_module[node]
                        begin_node_module = self.grad_fn_to_module[begin_node]

                        if begin_node_module in self.conv_fc_stride.keys():
                            stride = self.conv_fc_stride[begin_node_module]
                            if self.ignore_conv_fc_dep or stride < 0:
                                return
                            elif self.conv_fc_stride[begin_node_module] > 1:
                                dep.index_mapping = _FalttenIndexMapping(stride=stride)
                        if isinstance(begin_node_module, _ConcatOp):
                            dep.offset = begin_node_module.offset[path_id]

                        self.add_dependency(node_module, begin_node_module, dep)
                    if dep or rev_dep:
                        return

            if hasattr(node, 'next_functions'):
                path_id = 0
                for u in node.next_functions:
                    # print(u)
                    if u[0] is not None:
                        _recursively_detect_dependencies(u[0], path_id)
                        path_id += 1
        for begin_node in pending_nodes:
            _recursively_detect_dependencies(begin_node, 0)
