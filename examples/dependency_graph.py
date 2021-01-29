import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet18
import torch_pruning as tp
model = resnet18(pretrained=True)

# pruning according to L1 Norm
strategy = tp.strategy.L1Strategy() # or tp.strategy.RandomStrategy()
# build layer dependency for resnet18
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))
# get a pruning plan according to the dependency graph. idxs is the indices of pruned filters.
pruning_plan = DG.get_pruning_plan( model.conv1, tp.prune_conv, idxs=strategy(model.conv1.weight, amount=0.1) )
print(pruning_plan)
# execute this plan (prune the model)
pruning_plan.exec()