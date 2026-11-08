import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torch_pruning as tp


def test_importance_of_partial_group():
    # https://github.com/VainF/Torch-Pruning/issues/371
    model = resnet18().eval()
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 3, 224, 224))
    group = DG.get_pruning_group(model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9])
    for imp in [tp.importance.MagnitudeImportance(), tp.importance.GroupMagnitudeImportance(group_reduction="max"),
                tp.importance.GroupMagnitudeImportance(group_reduction="prod"), tp.importance.FPGMImportance(),
                tp.importance.BNScaleImportance(), tp.importance.LAMPImportance()]:
        score = imp(group)
        assert score.shape == (3,), type(imp).__name__

    full_group = DG.get_pruning_group(model.conv1, tp.prune_conv_out_channels, idxs=list(range(64)))
    imp = tp.importance.MagnitudeImportance(normalizer=None)
    assert torch.allclose(imp(full_group)[[2, 6, 9]], imp(group))
    # the scores follow the order of the given indices
    reordered = DG.get_pruning_group(model.conv1, tp.prune_conv_out_channels, idxs=[9, 2, 6])
    assert torch.allclose(imp(full_group)[[9, 2, 6]], imp(reordered))


def test_standardization_normalizer():
    # https://github.com/VainF/Torch-Pruning/issues/509
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 4, 3))
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 3, 8, 8))
    group = DG.get_pruning_group(model[0], tp.prune_conv_out_channels, idxs=list(range(8)))
    for name in ["standardization", "standarization"]: # the misspelled name is kept for backward compatibility
        score = tp.importance.GroupMagnitudeImportance(normalizer=name)(group)
        assert abs(score.min().item()) < 1e-6 and abs(score.max().item() - 1) < 1e-4


def test_fpgm_group_conv():
    # https://github.com/VainF/Torch-Pruning/issues/427
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(16, 32, 3, padding=1, groups=4), nn.ReLU(),
                          nn.Conv2d(32, 8, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10))
    x = torch.randn(1, 3, 16, 16)
    pruner = tp.pruner.MetaPruner(model, x, importance=tp.importance.FPGMImportance(), pruning_ratio=0.5,
                                  ignored_layers=[model[-1]])
    pruner.step()
    assert model[2].in_channels == 8 and model[2].out_channels == 16
    model(x)


def test_obdc_importance():
    # https://github.com/VainF/Torch-Pruning/issues/452
    # _prepare_model on a CNN without attention, layers with a bias
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 16, 3), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10))
    x = torch.randn(4, 3, 16, 16)
    imp = tp.importance.OBDCImportance(num_classes=10)
    pruner = tp.pruner.MetaPruner(model, x, importance=imp, pruning_ratio=0.5, ignored_layers=[model[-1]])
    model.zero_grad()
    imp._prepare_model(model, pruner)
    assert set(imp.modules) == {model[0], model[3]}
    F.cross_entropy(model(x), torch.randint(0, 10, (4,))).backward()
    imp.step()
    group = pruner.DG.get_pruning_group(model[0], tp.prune_conv_out_channels, idxs=[1, 5, 6])
    assert imp(group).shape == (3,)
    pruner.step()
    imp._rm_hooks(model)
    imp._clear_buffer()
    assert model[0].out_channels == 4 and model[3].out_channels == 8
    model(x)


if __name__ == "__main__":
    test_importance_of_partial_group()
    test_standardization_normalizer()
    test_fpgm_group_conv()
    test_obdc_importance()
