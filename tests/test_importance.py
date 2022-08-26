import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet18
import torch_pruning as tp
model = resnet18(pretrained=True)

# Global metrics

DG = tp.DependencyGraph()
example_inputs = torch.randn(1,3,224,224)
DG.build_dependency(model, example_inputs=example_inputs)
pruning_idxs = list( range( tp.utils.count_prunable_out_channels(model.conv1) ))
pruning_clique = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channel, idxs=pruning_idxs)

sensitivity_importance = tp.importance.SensitivityImportance(reduction='mean')
out = model(example_inputs)
loss = out.sum()
sen_imp = sensitivity_importance(loss, pruning_clique)
print(sen_imp)

magnitude_importance = tp.importance.MagnitudeImportance(p=2, reduction='mean')
mag_imp = magnitude_importance(pruning_clique)
print(mag_imp)

bn_scale_importance = tp.importance.BNScaleImportance(reduction='mean')
bn_imp = bn_scale_importance(pruning_clique)
print(bn_imp)



