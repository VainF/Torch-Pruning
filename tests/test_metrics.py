import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet18
import torch_pruning as tp
model = resnet18(pretrained=True)

# Global metrics
l2_norm_metric = tp.metric.NormMetric(p=1)
tp.set_global_metrics({"L2 Norm": l2_norm_metric})
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))
pruning_idxs =[0, 2, 6] 
pruning_plan = DG.get_pruning_plan( model.conv1, tp.prune_conv_out_channel, idxs=pruning_idxs)
print(pruning_plan)
pruning_plan.exec()

# Per-index metrics
l2_norm_metric.reduction = 'none'
pruning_plan = DG.get_pruning_plan( model.conv1, tp.prune_conv_out_channel, idxs=pruning_idxs )
print(pruning_plan)
pruning_plan.exec()