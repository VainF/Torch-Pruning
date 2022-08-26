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
    
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        x = torch.div(x, norm)
        return x * self.scale + self.bias

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
#
class MyPruningFn(tp.functional.structured.BasePruner):

    def prune(self, layer: CustomizedLayer, idxs: Sequence[int]) -> nn.Module: 
        keep_idxs = list(set(range(layer.in_dim)) - set(idxs))
        layer.in_dim = layer.in_dim-len(idxs)
        layer.scale = torch.nn.Parameter(layer.scale.data.clone()[keep_idxs])
        layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer
    
    @staticmethod
    def calc_nparams_to_prune(layer: CustomizedLayer, idxs: Sequence[int]) -> int: 
        nparams_to_prune = len(idxs) * 2
        return nparams_to_prune
        
my_pruning_fn = MyPruningFn()


model = FullyConnectedNet(128, 10, 256)
# pruning according to L1 Norm
strategy = tp.strategy.L1Strategy() # or tp.strategy.RandomStrategy()

DG = tp.DependencyGraph()
# Register your customized layer
DG.register_customized_layer(
    CustomizedLayer, 
    in_ch_pruning_fn=my_pruning_fn, # A function to prune channels/dimensions of input tensor
    out_ch_pruning_fn=my_pruning_fn, # A function to prune channels/dimensions of output tensor
    get_in_ch_fn=lambda l: l.in_dim,  # estimate the n_channel of layer input. Return None if the layer does not change tensor shape.
    get_out_ch_fn=lambda l: l.in_dim) # estimate the n_channel of layer output. Return None if the layer does not change tensor shape.

# Build dependency graph
DG.build_dependency(model, example_inputs=torch.randn(1,128))
# get a pruning group according to the dependency graph. idxs is the indices of pruned filters.
pruning_clique = DG.get_pruning_group( model.fc1, tp.prune_linear_out_channel, idxs=strategy(model.fc1.weight, amount=0.4) )
print(pruning_clique)

# execute this group (prune the model)
pruning_clique.exec()
print(model)
