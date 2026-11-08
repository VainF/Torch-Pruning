"""Node implementation for dependency graph."""
import torch.nn as nn
from .. import ops

class Node(object):
    """ Node of DepGraph. 
    """
    def __init__(self, module: nn.Module, grad_fn, name: str = None):
        # Connections in the computational graph
        self.inputs = []  # input nodes
        self.input_shapes = [] # self.input_shapes[i]: shape of the tensor received from self.inputs[i], None if unknown
        self.outputs = [] # output nodes
        self.output_nrs = [] # self.output_nrs[i]: which output of this op (e.g. which chunk of torch.split/unbind) self.outputs[i] consumes, None if unknown
        self.module = module # reference to torch.nn.Module
        self.grad_fn = grad_fn # grad_fn of nn.module output
        self._name = name # node name
        self.type = ops.module2type(module) # node type (enum), op.OPTYPE
        self.module_class = module.__class__ # class type of the module

        # For Dependency Modeling
        self.dependencies = []  # Adjacency List. It contains the dependencies to other nodes.
        self.enable_index_mapping = True # enable index mapping for torch.cat/split/chunck/...
        self.pruning_dim = -1 # pruning dimension for the module, whill be set dynamically by the Depdenency

    @property
    def name(self):
        if self._name is None:
            return str(self.module)
        else:
            fmt = self._name
            if self.type != ops.OPTYPE.PARAMETER:
                fmt += " ({})".format(str(self.module))
            return fmt

    def add_input(self, node, shape=None):
        self.inputs.append(node)
        self.input_shapes.append(shape)

    def add_output(self, node, output_nr=None):
        self.outputs.append(node)
        self.output_nrs.append(output_nr)

    def output_nr(self, i):
        """The output index of this op consumed by self.outputs[i]. Falls back to i if it was not recorded."""
        nr = self.output_nrs[i] if i < len(self.output_nrs) else None
        return i if nr is None else nr

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
        fmt += "\tenable_index_mapping={}, pruning_dim={}\n".format(
            self.enable_index_mapping, self.pruning_dim)
        fmt = "-" * 32 + "\n"
        return fmt
