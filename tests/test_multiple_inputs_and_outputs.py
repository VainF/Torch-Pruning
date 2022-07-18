import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp


class FullyConnectedNet(nn.Module):
    """https://github.com/VainF/Torch-Pruning/issues/21"""

    def __init__(self, input_sizes, output_sizes):
        super().__init__()

        self.fc1 = nn.Linear(input_sizes[0], output_sizes[0])
        self.fc2 = nn.Linear(input_sizes[1], output_sizes[1])
        self.fc3 = nn.Linear(sum(output_sizes), 1000)

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1(x1))
        x2 = F.relu(self.fc2(x2))
        x3 = F.relu(self.fc3(torch.cat([x1, x2], dim=1)))
        return x1, x2, x3


model = FullyConnectedNet([128, 64], [32, 32])

# pruning according to L1 Norm
strategy = tp.strategy.L1Strategy()  # or tp.strategy.RandomStrategy()

# Build dependency graph
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=[torch.randn(1, 128), torch.randn(1, 64)])

# get a pruning plan according to the dependency graph. idxs is the indices of pruned filters.
pruning_plan = DG.get_pruning_plan(
    model.fc1, tp.prune_linear_out_channel, idxs=strategy(model.fc1.weight, amount=0.4)
)
print(pruning_plan)

# execute this plan (prune the model)
pruning_plan.exec()

print(model)
