import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch_pruning as tp

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

model = Model()

# pruning according to L1 Norm
strategy = tp.strategy.L1Strategy() # or tp.strategy.RandomStrategy()
# build layer dependency for resnet18
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,1,28,28))
# get a pruning clique according to the dependency graph. idxs is the indices of pruned filters.
pruning_idxs = [0, 2, 6] #strategy(model.conv1.weight, amount=0.4) # or manually selected [0, 2, 6]
pruning_clique = DG.get_pruning_clique( model.fc2, tp.prune_linear_out_channel, idxs=pruning_idxs )
print(pruning_clique)
# execute this clique (prune the model)
if DG.check_pruning_clique(pruning_clique):
    pruning_clique.exec()