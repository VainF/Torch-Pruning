import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet18
import torch_pruning as pruning

model = resnet18(pretrained=True)
# build layer dependency for resnet18
DG = pruning.DependencyGraph( model, fake_input=torch.randn(1,3,224,224) )
# get a pruning plan according to the dependency graph of resnet18
pruning_plan = DG.get_pruning_plan( model.conv1, pruning.prune_conv, idxs=[2, 6, 9] )
print(pruning_plan)
# execute this plan
pruning_plan.exec()

print(model)