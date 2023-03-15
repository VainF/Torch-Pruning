import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet18
import torch_pruning as tp

model = resnet18(pretrained=True).eval()

# 1. build dependency graph for resnet18
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# 2. Select channels for pruning, here we prune the channels indexed by [2, 6, 9].
pruning_idxs = pruning_idxs=[2, 6, 9]
pruning_group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs )

# 3. prune all grouped layer that is coupled with model.conv1
if DG.check_pruning_group(pruning_group):
    pruning_group.exec()

print("Pruning Group:")
print(pruning_group)

print("After pruning:")
print(model)

groups = list(DG.get_all_groups())
print("Num groups: %d"%(len(groups)))

for g in groups:
    print(g)