import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet18
import torch_pruning as tp
model = resnet18()

# Global metrics

DG = tp.DependencyGraph()
example_inputs = torch.randn(1,3,224,224)
DG.build_dependency(model, example_inputs=example_inputs)
pruning_idxs = list( range( DG.get_out_channels(model.conv1) ))
pruning_group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs)

random_importance = tp.importance.RandomImportance()
rand_imp = random_importance(pruning_group)
print("Random: ", rand_imp)

magnitude_importance = tp.importance.MagnitudeImportance(p=1)
mag_imp = magnitude_importance(pruning_group)
print("L-1 Norm, Group Mean: ", mag_imp)

magnitude_importance = tp.importance.MagnitudeImportance(p=2)
mag_imp = magnitude_importance(pruning_group)
print("L-2 Norm, Group Mean: ", mag_imp)

magnitude_importance = tp.importance.MagnitudeImportance(p=2, group_reduction='sum')
mag_imp = magnitude_importance(pruning_group)
print("L-2 Norm, Group Sum: ", mag_imp)

magnitude_importance = tp.importance.MagnitudeImportance(p=2, group_reduction=None)
mag_imp = magnitude_importance(pruning_group)
print("L-2 Norm, No Reduction: ", mag_imp)

bn_scale_importance = tp.importance.BNScaleImportance()
bn_imp = bn_scale_importance(pruning_group)
print("BN Scaling, Group mean: ", bn_imp)

lamp_importance = tp.importance.LAMPImportance()
lamp_imp = lamp_importance(pruning_group)
print("LAMP: ", lamp_imp)


