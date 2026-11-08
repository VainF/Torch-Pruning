import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp


def test_glu_group():
    # https://github.com/VainF/Torch-Pruning/issues/366
    # GLU output channel j is computed from input channels j and j + C/2
    model = nn.Sequential(nn.Conv1d(96, 24, 3, padding=1), nn.GLU(1), nn.Conv1d(12, 48, 3, padding=2, dilation=2))
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 96, 100))
    group = DG.get_pruning_group(model[0], tp.prune_conv_out_channels, idxs=[3])
    layers = {dep.target.module: sorted(idxs) for dep, idxs in group}
    assert layers[model[0]] == [3, 15] and layers[model[2]] == [3], group
    group = DG.get_pruning_group(model[2], tp.prune_conv_in_channels, idxs=[0, 4])
    layers = {dep.target.module: sorted(idxs) for dep, idxs in group}
    assert layers[model[0]] == [0, 4, 12, 16], group


def test_glu_pruner():
    model = nn.Sequential(nn.Conv1d(96, 24, 3, padding=1), nn.GLU(1), nn.Conv1d(12, 48, 3, padding=2, dilation=2))
    x = torch.randn(1, 96, 100)
    pruner = tp.pruner.GroupNormPruner(model, x, importance=tp.importance.GroupMagnitudeImportance(), pruning_ratio=0.5)
    pruner.step()
    assert model[0].out_channels == 12 and model[2].in_channels == 6
    model(x)


def test_glu_iterative_pruning():
    model = nn.Sequential(nn.Conv1d(96, 24, 3, padding=1), nn.GLU(1), nn.Conv1d(12, 48, 3, padding=2, dilation=2),
                          nn.ReLU(), nn.Conv1d(48, 8, 1))
    x = torch.randn(2, 96, 50)
    pruner = tp.pruner.MetaPruner(model, x, importance=tp.importance.GroupMagnitudeImportance(), pruning_ratio=0.5,
                                  iterative_steps=3, ignored_layers=[model[-1]])
    for _ in range(3):
        pruner.step()
        model(x)
        assert model[0].out_channels == 2 * model[2].in_channels
    assert model[0].out_channels == 12 and model[2].in_channels == 6


class LinearGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(16, 64)
        self.down = nn.Linear(32, 16)
        self.head = nn.Linear(16, 4)

    def forward(self, x):
        return self.head(F.relu(self.down(F.glu(self.up(x), dim=-1))))


def test_glu_on_last_dim():
    model = LinearGLU()
    x = torch.randn(2, 7, 16)
    pruner = tp.pruner.MetaPruner(model, x, importance=tp.importance.GroupMagnitudeImportance(), pruning_ratio=0.25,
                                  global_pruning=True, ignored_layers=[model.head])
    pruner.step()
    assert model.up.out_features == 2 * model.down.in_features < 64
    model(x)


if __name__ == "__main__":
    test_glu_group()
    test_glu_pruner()
    test_glu_iterative_pruning()
    test_glu_on_last_dim()
