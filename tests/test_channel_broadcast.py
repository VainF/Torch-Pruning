import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp


class SpatialMask(nn.Module):
    """y = sigmoid(mask_conv(x)) * feat_conv(x), with a single-channel mask"""
    def __init__(self):
        super().__init__()
        self.feat = nn.Conv2d(3, 32, 3, padding=1)
        self.mask = nn.Conv2d(3, 1, 3, padding=1)
        self.out = nn.Conv2d(32, 8, 1)

    def forward(self, x):
        return self.out(torch.sigmoid(self.mask(x)) * self.feat(x))


def test_broadcast_mask_is_not_coupled():
    # https://github.com/VainF/Torch-Pruning/issues/510
    model = SpatialMask()
    x = torch.randn(1, 3, 8, 8)
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=x)
    group = DG.get_pruning_group(model.feat, tp.prune_conv_out_channels, idxs=[0, 5, 7])
    assert all(dep.target.module is not model.mask for dep, _ in group), group
    pruner = tp.pruner.MetaPruner(model, x, importance=tp.importance.GroupMagnitudeImportance(), pruning_ratio=0.5,
                                  ignored_layers=[model.out])
    pruner.step()
    assert model.feat.out_channels == 16 and model.mask.out_channels == 1
    model(x)


class Gate1d(nn.Module):
    """(N, 1, L) gate times (N, C, L) features"""
    def __init__(self):
        super().__init__()
        self.feat = nn.Conv1d(4, 32, 3, padding=1)
        self.gate = nn.Conv1d(4, 1, 3, padding=1)
        self.out = nn.Conv1d(32, 4, 1)

    def forward(self, x):
        return self.out(self.feat(x) * torch.sigmoid(self.gate(x)))


class TokenGate(nn.Module):
    """(N, L, 1) gate times (N, L, C) features"""
    def __init__(self):
        super().__init__()
        self.feat = nn.Linear(16, 32)
        self.gate = nn.Linear(16, 1)
        self.out = nn.Linear(32, 4)

    def forward(self, x):
        return self.out(self.feat(x) * torch.sigmoid(self.gate(x)))


def test_broadcast_gate_3d():
    for model, x in [(Gate1d(), torch.randn(2, 4, 20)), (TokenGate(), torch.randn(2, 5, 16))]:
        pruner = tp.pruner.MetaPruner(model, x, importance=tp.importance.GroupMagnitudeImportance(), pruning_ratio=0.5,
                                      ignored_layers=[model.out])
        pruner.step()
        feat_channels = model.feat.out_channels if isinstance(model.feat, nn.Conv1d) else model.feat.out_features
        gate_channels = model.gate.out_channels if isinstance(model.gate, nn.Conv1d) else model.gate.out_features
        assert feat_channels == 16 and gate_channels == 1, model
        model(x)


class SqueezeExcite(nn.Module):
    """the (N, C, 1, 1) scale is broadcast along H and W, not along the channels: it stays coupled"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc1 = nn.Conv2d(16, 4, 1)
        self.fc2 = nn.Conv2d(4, 16, 1)
        self.out = nn.Conv2d(16, 8, 1)

    def forward(self, x):
        y = self.conv(x)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(y, 1)))))
        return self.out(y * w)


def test_squeeze_excite_stays_coupled():
    model = SqueezeExcite()
    x = torch.randn(1, 3, 8, 8)
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=x)
    group = DG.get_pruning_group(model.conv, tp.prune_conv_out_channels, idxs=[2])
    assert any(dep.target.module is model.fc2 for dep, _ in group), group
    pruner = tp.pruner.MetaPruner(model, x, importance=tp.importance.GroupMagnitudeImportance(), pruning_ratio=0.5,
                                  ignored_layers=[model.out])
    pruner.step()
    assert model.conv.out_channels == model.fc2.out_channels == 8
    model(x)


def test_tensor_example_inputs_keep_batch_dim():
    # a (1, C, H, W) tensor used to be passed as model(*x), i.e. traced as an unbatched (C, H, W) input
    model = SpatialMask()
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 3, 8, 8))
    shape = tp._helpers.grad_fn_output_shape(DG.module2node[model.feat].grad_fn)
    if shape is not None: # needs a PyTorch release with autograd input metadata
        assert shape == (1, 32, 8, 8)


if __name__ == "__main__":
    test_broadcast_mask_is_not_coupled()
    test_broadcast_gate_3d()
    test_squeeze_excite_stays_coupled()
    test_tensor_example_inputs_keep_batch_dim()
