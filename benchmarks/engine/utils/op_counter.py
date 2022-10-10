
import torch
import numpy as np
import copy

def count_ops_and_params(model, example_inputs):
    flops = get_n_flops(model, example_inputs)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return flops, params

def get_n_flops(model, example_inputs, count_adds=True, idx_scale=None):
    '''Only count the FLOPs of conv and linear layers (no BN layers etc.). 
    Only count the weight computation (bias not included since it is negligible)
    '''

    model = copy.deepcopy(model)
    list_conv = []
    def conv_hook(self, input, output):
        flops = np.prod(self.weight.data.shape) * output.size(2) * output.size(3) / self.groups
        list_conv.append(flops)

    list_linear = []
    def linear_hook(self, input, output):
        flops = np.prod(self.weight.data.shape)
        list_linear.append(flops)

    def register_hooks(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            return
        for c in childrens:
            register_hooks(c)

    register_hooks(model)
    use_cuda = next(model.parameters()).is_cuda
    
    # forward
    model(example_inputs)
    # @mst (TODO): for SR network, there may be an extra argument for scale. Here set it to 2 to make it run normally. 
    # -- An ugly solution. Probably will be improved later.
    total_flops = (sum(list_conv) + sum(list_linear))
    if count_adds:
        total_flops *= 2
    return total_flops