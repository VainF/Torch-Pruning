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
        self.fc2 = nn.Linear(HIDDEN_UNITS, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        y_hat = self.fc2(x)
        return y_hat

model = FullyConnectedNet(128, 10, 256)

# pruning according to L1 Norm
strategy = tp.strategy.L1Strategy() # or tp.strategy.RandomStrategy()

# Build dependency graph
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,128))

# get a pruning plan according to the dependency graph. idxs is the indices of pruned filters.
pruning_plan = DG.get_pruning_plan( model.fc1, tp.prune_linear, idxs=strategy(model.fc1.weight, amount=0.4) )
print(pruning_plan)

# execute this plan (prune the model)
pruning_plan.exec()

print(model)
