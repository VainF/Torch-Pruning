import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from insightface import ResNet, IRBlock
from torchvision.models import resnet18
import torch_pruning as pruning

model = ResNet(IRBlock, [2, 2, 2, 2], use_se=False) 
#model = resnet18()

# build layer dependency for resnet18

DG = pruning.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,112,112))

# get a pruning plan according to the dependency graph. idxs is the indices of pruned filters.
pruning_plan = DG.get_pruning_plan( model.layer1[0].conv1, pruning.prune_conv, idxs=[2, 6, 9] )
print(pruning_plan)
# execute this plan (prune the model)
pruning_plan.exec()

# verify
with torch.no_grad():
    print( "Output: ", model( torch.randn( 1, 3, 112, 112 ) ).shape )

