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
# get a pruning clique according to the dependency graph. idxs is the indices of pruned filters.
pruning_idxs = [0, 2, 6] #strategy(model.conv1.weight, amount=0.4) # or manually selected [0, 2, 6]
pruning_clique = DG.get_pruning_clique( model.conv1, tp.prune_conv_out_channel, idxs=pruning_idxs )
print(pruning_clique)
# execute this clique (prune the model)
if DG.check_pruning_clique(pruning_clique):
    pruning_clique.exec()