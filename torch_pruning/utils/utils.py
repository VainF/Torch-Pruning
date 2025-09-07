"""Utility functions for torch-pruning."""

import torch
import torch.nn as nn

from ..ops import TORCH_BATCHNORM, TORCH_CONV, TORCH_LINEAR, TORCH_PRELU, module2type
from .op_counter import count_ops_and_params

@torch.no_grad()
def count_params(module):
    """Count the total number of parameters in a module.
    
    Args:
        module: PyTorch module to count parameters for.
        
    Returns:
        Total number of parameters.
    """
    return sum(p.numel() for p in module.parameters())


def flatten_as_list(obj):
    """Flatten nested structures into a flat list.
    
    Args:
        obj: Object to flatten (Tensor, list, tuple, dict, or other).
        
    Returns:
        Flattened list of objects.
    """
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

def draw_computational_graph(DG, save_as, title='Computational Graph', figsize=(16, 16), dpi=200, cmap=None):
    """Draw computational graph visualization.
    
    Args:
        DG: Dependency graph object.
        save_as: Path to save the figure.
        title: Title of the plot. Defaults to 'Computational Graph'.
        figsize: Figure size tuple. Defaults to (16, 16).
        dpi: Dots per inch. Defaults to 200.
        cmap: Colormap for the plot. Defaults to None.
        
    Returns:
        Tuple of (figure, axes) objects.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.style.use('bmh')
    n_nodes = len(DG.module2node)
    module2idx = {m: i for (i, m) in enumerate(DG.module2node.keys())}
    G = np.zeros((n_nodes, n_nodes))
    fill_value = 1
    
    for module, node in DG.module2node.items():
        for input_node in node.inputs:
            G[module2idx[input_node.module], module2idx[node.module]] = fill_value
            G[module2idx[node.module], module2idx[input_node.module]] = fill_value
        for out_node in node.outputs:
            G[module2idx[out_node.module], module2idx[node.module]] = fill_value
            G[module2idx[node.module], module2idx[out_node.module]] = fill_value
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(G, cmap=cmap if cmap is not None else plt.get_cmap('Blues'))
    
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    plt.savefig(save_as, dpi=dpi)
    return fig, ax


def draw_groups(DG, save_as, title='Group', figsize=(16, 16), dpi=200, cmap=None):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('bmh')
    n_nodes = 2*len(DG.module2node)
    node2idx = {m: i for (i, m) in enumerate(DG.module2node.values())}
    G = np.zeros((n_nodes, n_nodes))
    fill_value = 10
    for i, (module, node) in enumerate(DG.module2node.items()):
        pruning_fn = DG.get_pruner_of_module(module).prune_out_channels
        prunable_ch = DG.get_out_channels(module)
        if prunable_ch is None: continue
        group = DG.get_pruning_group(module, pruning_fn, list(range(prunable_ch)))
        grouped_idxs = []
        for dep, _ in group:
            source, target, trigger, handler = dep.source, dep.target, dep.trigger, dep.handler
            if DG.is_out_channel_pruning_fn(trigger):
                grouped_idxs.append(node2idx[source]*2+1)
            else:
                grouped_idxs.append(node2idx[source]*2)

            if DG.is_out_channel_pruning_fn(handler):
                grouped_idxs.append(node2idx[target]*2+1)
            else:
                grouped_idxs.append(node2idx[target]*2)
        grouped_idxs = list(set(grouped_idxs))
        for k1 in grouped_idxs:
            for k2 in grouped_idxs:
                G[k1, k2] = fill_value

    fig, ax = plt.subplots(figsize=(figsize))
    ax.imshow(G, cmap=cmap if cmap is not None else plt.get_cmap('Blues'))
    # plt.hlines(y=np.arange(0, n_nodes)+0.5, xmin=np.full(n_nodes, 0)-0.5, xmax=np.full(n_nodes, n_nodes)-0.5, color="#999999", linewidth=0.1)
    # plt.vlines(x=np.arange(0, n_nodes)+0.5, ymin=np.full(n_nodes, 0)-0.5, ymax=np.full(n_nodes, n_nodes)-0.5, color="#999999", linewidth=0.1)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    plt.savefig(save_as, dpi=dpi)
    return fig, ax


def draw_dependency_graph(DG, save_as, title='Group', figsize=(16, 16), dpi=200, cmap=None):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('bmh')
    n_nodes = len(DG.module2node)
    node2idx = {node: i for (i, node) in enumerate(DG.module2node.values())}
    G = np.zeros((2*n_nodes, 2*n_nodes))
    fill_value = 10
    for module, node in DG.module2node.items():
        for dep in node.dependencies:
            trigger = dep.trigger
            handler = dep.handler
            source = dep.source
            target = dep.target

            if DG.is_out_channel_pruning_fn(trigger):
                G[2*node2idx[source]+1, 2*node2idx[target]] = fill_value
            else:
                G[2*node2idx[source], 2*node2idx[target]+1] = fill_value

        pruner = DG.get_pruner_of_module(module)
        if pruner.prune_out_channels == pruner.prune_in_channels:
            G[2*node2idx[node], 2*node2idx[node]+1] = fill_value

    fig, ax = plt.subplots(figsize=(figsize))
    ax.imshow(G, cmap=cmap if cmap is not None else plt.get_cmap('Blues'))
    # plt.hlines(y=np.arange(0, 2*n_nodes)+0.5, xmin=np.full(2*n_nodes, 0)-0.5, xmax=np.full(2*n_nodes, 2*n_nodes)-0.5, color="#999999", linewidth=0.05)
    # plt.vlines(x=np.arange(0, 2*n_nodes)+0.5, ymin=np.full(2*n_nodes, 0)-0.5, ymax=np.full(2*n_nodes, 2*n_nodes)-0.5, color="#999999", linewidth=0.05)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    plt.savefig(save_as, dpi=dpi)
    return fig, ax

class print_tool():
    _before_pruning = None

    @staticmethod
    def before_pruning(model):
        """Record the model structure before pruning.
        
        Args:
            model: The model to record.
        """
        print_tool._before_pruning = str(model)

    @staticmethod
    def after_pruning(model, do_print=True):
        """Compare and display model structure after pruning.
        
        Args:
            model: The model after pruning.
            do_print: Whether to print the comparison. Defaults to True.
            
        Returns:
            String representation of the comparison.
            
        Raises:
            ValueError: If before_pruning was not called first.
        """
        if print_tool._before_pruning is None:
            raise ValueError(
                "Please call PrintTool.before_pruning(model) to record the original model."
            )
        
        _after_pruning = str(model)
        model_str = ""
        
        # Iterate all lines and compare
        for line1, line2 in zip(print_tool._before_pruning.split('\n'), _after_pruning.split('\n')):
            if line1 != line2:
                line2 = line2.lstrip(' ')  # remove the leading spaces
                model_str += f"{line1} => {line2}\n"
            else:
                model_str += line1 + '\n'
        
        if do_print:
            print(model_str)
        
        print_tool._before_pruning = None
        return model_str

                

    

    