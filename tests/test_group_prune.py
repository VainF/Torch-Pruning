import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet18
import torch_pruning as tp

def test_depgraph():
    model = resnet18(pretrained=True).eval()
    # 1. build dependency graph for resnet18
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))
    # 2. Select channels for pruning, here we prune the channels indexed by [2, 6, 9].
    pruning_idxs = pruning_idxs=[2, 6, 9]
    pruning_group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs )
    pruning_group.prune()
    affected_weights1 = []
    for dep, _ in pruning_group:
        module = dep.target.module
        if hasattr(module, 'weight'):
            affected_weights1.append(module.weight.detach())
        if hasattr(module, 'bias') and module.bias is not None:
            affected_weights1.append(module.bias.detach())

    model = resnet18(pretrained=True).eval()
    # 1. build dependency graph for resnet18
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))
    # 2. Select channels for pruning
    pruning_idxs = pruning_idxs=[1, 2, 3, 4] # we will replace it with [2,6,9]
    pruning_group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs )
    pruning_group.prune([2,6,9])
    affected_weights2 = []
    for dep, _ in pruning_group:
        module = dep.target.module
        if hasattr(module, 'weight'):
            affected_weights2.append(module.weight.detach())
        if hasattr(module, 'bias') and module.bias is not None:
            affected_weights2.append(module.bias.detach())

    for w1, w2 in zip(affected_weights1, affected_weights2):
        assert torch.allclose(w1, w2)

if __name__=='__main__':
    test_depgraph()