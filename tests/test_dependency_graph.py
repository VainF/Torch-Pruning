import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet18
import torch_pruning as tp
model = resnet18(pretrained=True)

# build layer dependency for resnet18
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# get a pruning group according to the dependency graph. idxs is the indices of pruned filters.
pruning_idxs = [0, 2, 6]
pruning_group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs )
print(pruning_group)

# execute this group (prune the model)
if DG.check_pruning_group(pruning_group):
    pruning_group.exec()

print("The pruned model: \n", model)
print("Output:", model(torch.randn(1,3,224,224)).shape)