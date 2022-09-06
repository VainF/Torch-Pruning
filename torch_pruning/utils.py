from .ops import TORCH_CONV, TORCH_BATCHNORM, TORCH_PRELU, TORCH_LINEAR
from .ops import module2type
import torch
import thop
import torch.nn as nn


#def count_prunable_params_of_modules(module):
#    if isinstance(module, (TORCH_CONV, TORCH_LINEAR)):
#        num_params = module.weight.numel()
#        if module.bias is not None:
#            num_params += module.bias.numel()
#        return num_params
#    elif isinstance(module, TORCH_BATCHNORM):
#        num_params = module.running_mean.numel() + module.running_var.numel()
#        if module.affine:
#            num_params += module.weight.numel() + module.bias.numel()
#        return num_params
#    elif isinstance(module, TORCH_PRELU):
#        if len(module.weight) == 1:
#            return 0
#        else:
#            return module.weight.numel
#    else:
#        return 0


#def count_prunable_in_channels(module):
#    if isinstance(module, TORCH_CONV):
#        return module.weight.shape[1]
#    elif isinstance(module, TORCH_LINEAR):
#        return module.in_features
#    elif isinstance(module, TORCH_BATCHNORM):
#        return module.num_features
#    elif isinstance(module, TORCH_PRELU):
#        if len(module.weight) == 1:
#            return 0
#        else:
#            return len(module.weight)
#    else:
#        return 0
#
#
#def count_prunable_out_channels(module):
#    if isinstance(module, TORCH_CONV):
#        return module.weight.shape[0]
#    elif isinstance(module, TORCH_LINEAR):
#        return module.out_features
#    elif isinstance(module, TORCH_BATCHNORM):
#        return module.num_features
#    elif isinstance(module, TORCH_PRELU):
#        if len(module.weight) == 1:
#            return 0
#        else:
#            return len(module.weight)
#    else:
#        return 0


def count_params(module):
    return sum([p.numel() for p in module.parameters()])


def count_ops_and_params(model, input_size=None, example_inputs=None, return_macs=False, device=None):
    if example_inputs is None:
        example_inputs = torch.randn(*input_size)
    if device is not None:
        example_inputs = example_inputs.to(device)
    macs, params = thop.profile(
        model, inputs=(example_inputs, ), verbose=False)
    if return_macs:
        return macs, params
    else:
        return 2*macs, params  # FLOP = 2*MACs

class GConv(nn.Module):
    def __init__(self, gconv):
        super(GConv, self).__init__()
        self.groups = gconv.groups
        self.convs = nn.ModuleList()
        oc_size = gconv.out_channels // self.groups
        ic_size = gconv.in_channels // self.groups
        for g in range(self.groups):
            self.convs.append(
                nn.Conv2d(
                    in_channels=oc_size,
                    out_channels=ic_size,
                    kernel_size=gconv.kernel_size,
                    stride=gconv.stride,
                    padding=gconv.padding,
                    dilation=gconv.dilation,
                    groups=1,
                    bias=gconv.bias is not None,
                    padding_mode=gconv.padding_mode,
                )
            )
        # copy parameters
        group_size = gconv.out_channels // self.groups
        gconv_weight = gconv.weight
        for (i, conv) in enumerate(self.convs):
            conv.weight.data = gconv_weight.data[oc_size *
                                                 i: oc_size * (i + 1)]
            if gconv.bias is not None:
                conv.bias.data = gconv.bias.data[oc_size *
                                                 i: oc_size * (i + 1)]

    def forward(self, x):
        split_sizes = [conv.in_channels for conv in self.convs]
        xs = torch.split(x, split_sizes, dim=1)
        out = torch.cat([conv(xi)
                        for (conv, xi) in zip(self.convs, xs)], dim=1)
        return out


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


def gconv2convs(module):
    new_module = module
    if (
        isinstance(module, nn.Conv2d)
        and module.groups > 1
        and module.groups != module.in_channels
    ):
        new_module = GConv(module)
    for name, child in module.named_children():
        new_module.add_module(name, gconv2convs(child))
    return new_module


def draw_computational_graph(DG, save_as, title='Dependency Graph', figsize=(16, 16), dpi=200, cmap=None):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('bmh')
    n_nodes = len(DG.module2node)
    print(n_nodes)
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
        fns = DG.PRUNING_FN[module2type(module)]
        if fns[0] == fns[1]:
            G[module2idx[node.module], module2idx[node.module]] = fill_value
    fig, ax = plt.subplots(figsize=(figsize))
    ax.imshow(G, cmap=cmap if cmap is not None else plt.get_cmap('Blues'))
    # plt.hlines(y=np.arange(0, n_nodes)+0.5, xmin=np.full(n_nodes, 0)-0.5, xmax=np.full(n_nodes, n_nodes)-0.5, color="#444444", linewidth=0.1)
    # plt.vlines(x=np.arange(0, n_nodes)+0.5, ymin=np.full(n_nodes, 0)-0.5, ymax=np.full(n_nodes, n_nodes)-0.5, color="#444444", linewidth=0.1)
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
        group = DG.get_pruning_group(module, DG.PRUNING_FN[module2type(
            module)][1], list(range(count_prunable_out_channels(module))))
        grouped_idxs = []
        for dep, _ in group:
            source, target, trigger, handler = dep.source, dep.target, dep.trigger, dep.handler
            if trigger in DG.out_channel_pruners:
                grouped_idxs.append(node2idx[source]*2+1)
            else:
                grouped_idxs.append(node2idx[source]*2)

            if handler in DG.out_channel_pruners:
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

            if trigger in DG.out_channel_pruners:
                G[2*node2idx[source]+1, 2*node2idx[target]] = fill_value
            else:
                G[2*node2idx[source], 2*node2idx[target]+1] = fill_value

        fns = DG.PRUNING_FN[module2type(module)]
        if fns[0] == fns[1]:
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


# def draw_dependency_graph(DG, save_as, title='Dependency Graph', figsize=(16, 16), dpi=200, cmap=None):
#    import numpy as np
#    import matplotlib.pyplot as plt
#    n_nodes = len(DG.module2node)
#    module2idx = { m: i for (i, m) in enumerate(DG.module2node.keys()) }
#    G = np.zeros((n_nodes, n_nodes))
#    fill_value = 1
#    for module, node in DG.module2node.items():
#        for input_node in node.inputs:
#            G[ module2idx[input_node.module], module2idx[node.module] ] = fill_value
#            G[ module2idx[node.module], module2idx[input_node.module] ] = fill_value
#        for out_node in node.outputs:
#            G[ module2idx[out_node.module], module2idx[node.module] ] = fill_value
#            G[ module2idx[node.module], module2idx[out_node.module] ] = fill_value
#        fns = DG.PRUNING_FN[module2type(module)]
#        if fns[0] == fns[1]:
#            G[ module2idx[node.module], module2idx[node.module] ] = fill_value
#    fig, ax = plt.subplots(figsize=(figsize))
#    ax.imshow(G, cmap=cmap if cmap is not None else plt.get_cmap('Blues'))
#    plt.hlines(y=np.arange(0, n_nodes)+0.5, xmin=np.full(n_nodes, 0)-0.5, xmax=np.full(n_nodes, n_nodes)-0.5, color="#444444", linewidth=0.1)
#    plt.vlines(x=np.arange(0, n_nodes)+0.5, ymin=np.full(n_nodes, 0)-0.5, ymax=np.full(n_nodes, n_nodes)-0.5, color="#444444", linewidth=0.1)
#    ax.set_title(title)
#    fig.tight_layout()
#    plt.savefig(save_as, dpi=dpi)
#    return fig, ax
#
# def draw_groups(DG, save_as, title='Group', figsize=(16, 16), dpi=200, cmap=None):
#    import numpy as np
#    import matplotlib.pyplot as plt
#    n_nodes = len(DG.module2node)
#    module2idx = { m: i for (i, m) in enumerate(DG.module2node.keys()) }
#    G = np.zeros((n_nodes, n_nodes))
#    fill_value=1
#    for i, (module, node) in enumerate(DG.module2node.items()):
#        if not isinstance(module, tuple(DG.PRUNABLE_MODULES)):
#            continue
#        group = DG.get_pruning_group(module, DG.PRUNING_FN[module2type(module)][1], list(range(count_prunable_out_channels(module))))
#        for dep, _ in group:
#            G[ module2idx[module], module2idx[dep.target.module] ] = fill_value
#    fig, ax = plt.subplots(figsize=(figsize))
#    ax.imshow(G, cmap=cmap if cmap is not None else plt.get_cmap('Blues'))
#    plt.hlines(y=np.arange(0, n_nodes)+0.5, xmin=np.full(n_nodes, 0)-0.5, xmax=np.full(n_nodes, n_nodes)-0.5, color="#444444", linewidth=0.1)
#    plt.vlines(x=np.arange(0, n_nodes)+0.5, ymin=np.full(n_nodes, 0)-0.5, ymax=np.full(n_nodes, n_nodes)-0.5, color="#444444", linewidth=0.1)
#    ax.set_title(title)
#    fig.tight_layout()
#    plt.savefig(save_as, dpi=dpi)
#    return fig, ax
