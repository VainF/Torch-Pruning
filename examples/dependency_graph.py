import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet18
import torch_pruning as pruning
model = resnet18(pretrained=True)
# build layer dependency for resnet18
DG = pruning.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))
# get a pruning plan according to the dependency graph. idxs is the indices of pruned filters.
pruning_plan = DG.get_pruning_plan( model.conv1, pruning.prune_conv, idxs=[2, 6, 9] )
print(pruning_plan)
# execute this plan (prune the model)
pruning_plan.exec()

# verify
with torch.no_grad():
    print( "Output: ", model( torch.randn( 1, 3, 224, 224 ) ).shape )

