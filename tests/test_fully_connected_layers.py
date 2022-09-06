import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp

class FullyConnectedNet(nn.Module):
    """https://github.com/VainF/Torch-Pruning/issues/21"""
    def __init__(self, input_size, num_classes, HIDDEN_UNITS):
        super().__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS)
        self.fc3 = nn.Linear(HIDDEN_UNITS, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        skip=x
        x = F.relu(self.fc2(x))
        x = x+skip 
        x = self.fc3(x)
        return x

model = FullyConnectedNet(128, 10, 256)

# Build dependency graph
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,128))

# get a pruning group according to the dependency graph.
pruning_group = DG.get_pruning_group( model.fc1, tp.prune_linear_out_channels, idxs=[0, 4, 6] )
print(pruning_group)

# execute the group (prune the model)
pruning_group.exec()
print(model)

print("The pruned model: \n", model)
print("Output:", model(torch.randn(1,128)).shape)