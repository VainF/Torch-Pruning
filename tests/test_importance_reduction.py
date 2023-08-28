import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet18
import torch_pruning as tp
model = resnet18()

# Global metrics
def test_imp():
    DG = tp.DependencyGraph()
    example_inputs = torch.randn(1,3,224,224)
    DG.build_dependency(model, example_inputs=example_inputs)
    pruning_idxs = list( range( DG.get_out_channels(model.conv1) ))
    pruning_group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs)

    random_importance = tp.importance.RandomImportance()
    rand_imp = random_importance(pruning_group)
    print("Random: ", rand_imp)

    magnitude_importance = tp.importance.MagnitudeImportance(p=1, group_reduction=None, normalizer=None, bias=True)
    mag_imp_raw = magnitude_importance(pruning_group)
    print("L-1 Norm, No Reduction: ", mag_imp_raw)

    magnitude_importance = tp.importance.MagnitudeImportance(p=1, normalizer=None, bias=True)
    mag_imp = magnitude_importance(pruning_group)
    print("L-1 Norm, Group Mean: ", mag_imp)
    assert torch.allclose(mag_imp, mag_imp_raw.mean(0))

    magnitude_importance = tp.importance.MagnitudeImportance(p=2, group_reduction=None, normalizer=None, bias=True)
    mag_imp_raw = magnitude_importance(pruning_group)
    print("L-2 Norm, No Reduction: ", mag_imp_raw)

    magnitude_importance = tp.importance.MagnitudeImportance(p=2, normalizer=None, bias=True)
    mag_imp = magnitude_importance(pruning_group)
    print("L-2 Norm, Group Mean: ", mag_imp)
    assert torch.allclose(mag_imp, mag_imp_raw.mean(0))

    magnitude_importance = tp.importance.MagnitudeImportance(p=2, group_reduction='sum', normalizer=None, bias=True)
    mag_imp = magnitude_importance(pruning_group)
    print("L-2 Norm, Group Sum: ", mag_imp)
    assert torch.allclose(mag_imp, mag_imp_raw.sum(0))

    magnitude_importance = tp.importance.MagnitudeImportance(p=2, group_reduction='max', normalizer=None, bias=True)
    mag_imp = magnitude_importance(pruning_group)
    print("L-2 Norm, Group Max: ", mag_imp)
    assert torch.allclose(mag_imp, mag_imp_raw.max(0)[0])

    magnitude_importance = tp.importance.MagnitudeImportance(p=2, group_reduction='gate', normalizer=None, bias=True)
    mag_imp = magnitude_importance(pruning_group)
    print("L-2 Norm, Group Gate: ", mag_imp)
    assert torch.allclose(mag_imp, mag_imp_raw[-1])

    magnitude_importance = tp.importance.MagnitudeImportance(p=2, group_reduction='prod', normalizer=None, bias=True)
    mag_imp = magnitude_importance(pruning_group)
    print("L-2 Norm, Group Prod: ", mag_imp)
    print(mag_imp,  torch.prod(mag_imp_raw, dim=0))
    assert torch.allclose(mag_imp, torch.prod(mag_imp_raw, dim=0))

    bn_scale_importance = tp.importance.BNScaleImportance(normalizer=None)
    bn_imp = bn_scale_importance(pruning_group)
    print("BN Scaling, Group mean: ", bn_imp)   

    lamp_importance = tp.importance.LAMPImportance(bias=True)
    lamp_imp = lamp_importance(pruning_group)
    print("LAMP: ", lamp_imp)

    model(example_inputs).sum().backward()
    taylor_importance = tp.importance.TaylorImportance(normalizer='mean', bias=True)
    taylor_imp = taylor_importance(pruning_group)
    print("Taylor Importance", taylor_imp)  

if __name__=='__main__':
    test_imp()