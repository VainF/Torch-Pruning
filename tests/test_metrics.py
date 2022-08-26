import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet18
import torch_pruning as tp
model = resnet18(pretrained=True)

# Global metrics
l2_norm_metric = tp.metric.NormMetric(p=1)
tp.add_global_metrics(name="L2 Norm", metric_fn=l2_norm_metric)

DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))
pruning_idxs =[0, 2, 6] 
pruning_clique = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channel, idxs=pruning_idxs)
print(pruning_clique)
pruning_clique.exec()

# Per-index metrics
l2_norm_metric.reduction = 'none'
pruning_clique = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channel, idxs=pruning_idxs )
print(pruning_clique)
pruning_clique.exec()