"""Regression coverage for https://github.com/VainF/Torch-Pruning/issues/532."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torch_pruning as tp
from torch_pruning import ops
from torch_pruning.dependency import index_mapping


@pytest.mark.parametrize("shape,dim,is_channel", [
    ((2, 3, 4, 5), -1, False),
    ((2, 3, 4, 5), -2, False),
    ((2, 3, 4, 5), 3, False),
    ((2, 3, 4, 5), -3, True),
    ((2, 3, 4, 5), 1, True),
    ((2, 3), -1, True),
])
def test_real_autograd_concat_dimension(shape, dim, is_channel, monkeypatch):
    x = torch.randn(shape, requires_grad=True)
    grad_fn = torch.cat((x, x), dim=dim).grad_fn
    node = SimpleNamespace(
        type=ops.OPTYPE.CONCAT, grad_fn=grad_fn,
        module=SimpleNamespace(concat_sizes=[3, 3], offsets=None),
        inputs=[], dependencies=[],
    )
    # Make channel inference permissive: the dimension guard must reject
    # non-channel concatenation before it can install spurious offsets.
    monkeypatch.setattr(index_mapping.shape_infer, "infer_in_channels_recursively",
                        lambda *args, **kwargs: 6)
    index_mapping.update_concat_index_mapping(None, node)
    assert node.module.offsets == ([0, 3, 6] if is_channel else None)


@pytest.mark.parametrize("dim", [-1, (1 << 64) - 1])
def test_signed_and_unsigned_negative_dim_without_metadata(dim):
    node = SimpleNamespace(type=ops.OPTYPE.CONCAT,
                           grad_fn=SimpleNamespace(_saved_dim=dim))
    # Older autograd metadata may omit the output shape. No mapping should be
    # inferred from a negative dimension whose rank is unknown.
    index_mapping.update_concat_index_mapping(None, node)


@pytest.mark.parametrize("dim", [1, -1])
def test_linear_channel_concat_still_prunes(dim):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(4, 3)
            self.right = nn.Linear(4, 3)
            self.out = nn.Linear(6, 2)

        def forward(self, x):
            return self.out(torch.cat((self.left(x), self.right(x)), dim=dim))

    model = Model()
    x = torch.randn(2, 4)
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=x)
    group = graph.get_pruning_group(model.left, tp.prune_linear_out_channels, idxs=[1])
    group.prune()
    assert model.left.out_features == 2
    assert model.right.out_features == 3
    assert model.out.in_features == 5
    assert model(x).shape == (2, 2)
