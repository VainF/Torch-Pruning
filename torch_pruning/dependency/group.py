"""Group class implementation for handling pruning operations."""
import warnings
import typing
from .. import _helpers, ops
from .dependency import Dependency

class Group(object):
    """Group is the basic unit for pruning. It contains a list of dependencies and their corresponding indices.  
    group := [ (Dep1, Indices1), (Dep2, Indices2), ..., (DepK, IndicesK) ]

    Example: 

    For a simple network Conv2d(2, 4) -> BN(4) -> Relu, we have:
    group1 := [ (Conv2d -> BN, [0, 1, 2, 3]), (BN -> Relu, [0, 1, 2, 3]) ]
    There are 4 prunable elements, i.e., 4 channels in Conv2d.

    The indices do not need to be full and can be a subset of the prunable elements.
    For instance, if we want to prune the first 2 channels, we have:
    group2 := [ (Conv2d -> BN, [0, 1]), (BN -> Relu, [0, 1]) ]

    When combined with tp.importance, we can compute the importance of corresponding channels.
    imp_1 = importance(group1) # len(imp_1)=4
    imp_2 = importance(group2) # len(imp_2)=2
    
    For importance estimation, we should craft a group with full indices just like group1.
    For pruning, we need to craft a new group with the to-be-pruned indices like group2.
    """
    def __init__(self):
        self._group = list()
        self._DG = None # the dependency graph that this group belongs to

    def prune(self, idxs=None, record_history=True):
        """Prune all coupled layers in the group, acording to the specified indices.
        """
        if idxs is not None: # prune the group with user-specified indices
            module = self._group[0].dep.target.module
            pruning_fn = self._group[0].dep.handler
            new_group = self._DG.get_pruning_group(module, pruning_fn, idxs) # create a new group with the specified indices
            new_group.prune()
        else: # prune the group with the pre-defined indices
            for dep, idxs in self._group:
                if dep.target.type == ops.OPTYPE.PARAMETER: # for nn.Parameter, we will craft a new nn.Parameter and have to update all depdencies
                    # prune unwrapped nn.Parameter
                    old_parameter = dep.target.module
                    name = self._DG._param_to_name[old_parameter]
                    self._DG._param_to_name.pop(old_parameter)
                    pruned_parameter = dep(idxs)
                    path = name.split('.')
                    # fetch the the parent module of the parameter
                    module = self._DG.model
                    for p in path[:-1]:
                        module = getattr(module, p)
                    setattr(module, path[-1], pruned_parameter)
                    # update the dependency graph with the new parameter
                    self._DG._param_to_name[pruned_parameter] = name
                    self._DG.module2node[pruned_parameter] = self._DG.module2node.pop(old_parameter)
                    self._DG.module2node[pruned_parameter].module = pruned_parameter           
                else: # in most cases, we can directly prune the module
                    dep(idxs)
        
        if record_history: # record the pruning history
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
        """Add a new dependency and merge the indices if the dependency already exists.
        """
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
            fmt += "[{}] {}, len(idxs)={}\n".format(i, dep, len(idxs))
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
