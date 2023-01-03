import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp
from typing import Sequence

############
# Customize your layer
#
class CustomizedLayer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.scale = nn.Parameter(torch.Tensor(self.in_dim))
        self.bias = nn.Parameter(torch.Tensor(self.in_dim))
        self.fc = nn.Linear(self.in_dim, self.in_dim)
    
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        x = torch.div(x, norm)
        return self.fc(x * self.scale + self.bias)

    def __repr__(self):
        return "CustomizedLayer(in_dim=%d)"%(self.in_dim)

class FullyConnectedNet(nn.Module):
    """https://github.com/VainF/Torch-Pruning/issues/21"""
    def __init__(self, input_size, num_classes, HIDDEN_UNITS):
        super().__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_UNITS)
        self.customized_layer = CustomizedLayer(HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.customized_layer(x)
        y_hat = self.fc2(x)
        return y_hat

############################
# Implement your pruning function for the customized layer
# You should implement the following class fucntions:
# 1. prune_out_channels
# 2. prune_in_channels
# 3. get_out_channels
# 4. get_in_channels

class MyPruner(tp.pruner.BasePruningFunc):

    def prune_out_channels(self, layer: CustomizedLayer, idxs: Sequence[int]) -> nn.Module: 
        keep_idxs = list(set(range(layer.in_dim)) - set(idxs))
        keep_idxs.sort()
        layer.in_dim = layer.in_dim-len(idxs)
        layer.scale = torch.nn.Parameter(layer.scale.data.clone()[keep_idxs])
        layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        tp.prune_linear_in_channels(layer.fc, idxs)
        tp.prune_linear_out_channels(layer.fc, idxs)
        return layer

    def get_out_channels(self, layer):
        return self.in_dim
    
    # identical functions
    prune_in_channels = prune_out_channels
    get_in_channels = get_out_channels
        
model = FullyConnectedNet(128, 10, 256)

DG = tp.DependencyGraph()

# 1. Register your customized layer
my_pruner = MyPruner()
DG.register_customized_layer(
    CustomizedLayer, 
    my_pruner)

# 2. Build dependency graph
DG.build_dependency(model, example_inputs=torch.randn(1,128))

# 3. get a pruning group according to the dependency graph. idxs is the indices of pruned filters.
pruning_group = DG.get_pruning_group( model.fc1, tp.prune_linear_out_channels, idxs=[0, 1, 6] )
print(pruning_group)

# 4. execute this group (prune the model)
pruning_group.exec()
print("The pruned model:\n", model)
print("Output: ", model(torch.randn(1,128)).shape)

assert model.fc1.out_features==253 and model.customized_layer.in_dim==253 and model.fc2.in_features==253
