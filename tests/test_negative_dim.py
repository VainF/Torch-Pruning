import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp


def test_normalize_saved_dim():
    # Recent PyTorch releases expose negative dims saved by autograd as unsigned 64-bit integers
    x = torch.randn(2, 3, 4, 6, requires_grad=True)
    saved = torch.cat((x[..., :2], x[..., 2:]), dim=-1).grad_fn._saved_dim
    assert tp._helpers.normalize_saved_dim(saved, 4) == 3
    assert tp._helpers.normalize_saved_dim(2**64 - 1, 4) == 3
    assert tp._helpers.normalize_saved_dim(-1, 4) == 3
    assert tp._helpers.normalize_saved_dim(-1) == -1
    assert tp._helpers.normalize_saved_dim(1, 4) == 1


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RoPEScore(nn.Module):
    """Rotary embedding on attention heads. The width of the concat output cannot be inferred from its consumers."""
    def __init__(self, dim=64, heads=4, seq=8):
        super().__init__()
        self.q_proj = nn.Linear(16, dim)
        self.heads, self.head_dim = heads, dim // heads
        pos = torch.arange(seq).float()[:, None] * torch.arange(self.head_dim).float()[None, :] / self.head_dim
        self.register_buffer("cos", pos.cos())
        self.register_buffer("sin", pos.sin())

    def forward(self, x):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        q = q * self.cos + rotate_half(q) * self.sin
        return (q @ q.transpose(-1, -2)).softmax(-1).sum()


def test_rope_concat_on_head_dim():
    # https://github.com/VainF/Torch-Pruning/issues/532
    # torch.cat(..., dim=-1) on a (B, H, N, D) tensor must not shift channel indices. It did when -1 was read
    # as 2**64-1, and the shifted indices grew forever along the q -> rotate_half -> q cycle.
    model = RoPEScore()
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(2, 8, 16))
    concat = [n for n in DG.module2node.values() if n.type == tp.ops.OPTYPE.CONCAT]
    assert len(concat) == 1 and concat[0].module.offsets is None
    group = DG.get_pruning_group(model.q_proj, tp.prune_linear_out_channels, idxs=[2, 6, 9])
    for dep, idxs in group:
        assert max(idxs) < 64, group


class RoPEAttention(nn.Module):
    def __init__(self, dim=64, heads=4, seq=8):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = [nn.Linear(dim, dim) for _ in range(4)]
        pos = torch.arange(seq).float()[:, None] * torch.arange(self.head_dim).float()[None, :] / self.head_dim
        self.register_buffer("cos", pos.cos())
        self.register_buffer("sin", pos.sin())

    def forward(self, x):
        B, N, _ = x.shape
        q, k, v = [p(x).view(B, N, self.heads, -1).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj)]
        q = q * self.cos + rotate_half(q) * self.sin
        k = k * self.cos + rotate_half(k) * self.sin
        o = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(o.transpose(1, 2).reshape(B, N, -1))


class RoPENet(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.embed = nn.Linear(16, dim)
        self.attn = RoPEAttention(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.embed(x)
        x = x + self.attn(x)
        return self.head(x.mean(1))


def test_rope_attention_head_pruning():
    model = RoPENet()
    x = torch.randn(2, 8, 16)
    attn = model.attn
    num_heads = {attn.q_proj: 4, attn.k_proj: 4, attn.v_proj: 4}
    pruner = tp.pruner.MetaPruner(model, x, importance=tp.importance.GroupMagnitudeImportance(), pruning_ratio=0.5,
                                  num_heads=num_heads, prune_num_heads=True, prune_head_dims=False, head_pruning_ratio=0.5,
                                  ignored_layers=[model.head])
    pruner.step()
    assert attn.q_proj.out_features == attn.k_proj.out_features == attn.v_proj.out_features == 32
    assert attn.o_proj.in_features == 32
    attn.heads = pruner.num_heads[attn.q_proj]
    assert attn.heads == 2
    model(x)


class WideDeep(nn.Module):
    def __init__(self, input_dim=50):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 1))
        self.linear = nn.Linear(input_dim, 3)
        self.lin2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.concat([self.linear(x), self.mlp(x)], dim=-1)
        return torch.sigmoid(self.lin2(x))


def test_concat_last_dim_2d():
    # https://github.com/VainF/Torch-Pruning/issues/481
    # dim=-1 is the feature dim of a (N, C) tensor; PyTorch releases that expose it as -1 lost the concat offsets
    model = WideDeep()
    x = torch.randn(4, 50)
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=x)
    group = DG.get_pruning_group(model.linear, tp.prune_linear_out_channels, idxs=[0, 2])
    layers = {dep.target.module: idxs for dep, idxs in group}
    assert model.mlp[6] not in layers, group
    assert sorted(layers[model.lin2]) == [0, 2], group
    group.prune()
    assert model.linear.out_features == 1 and model.lin2.in_features == 2
    model(x)


class ChunkWidth(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 16, 3, padding=1)
        self.b = nn.Conv2d(16, 8, 1)

    def forward(self, x):
        left, right = self.a(x).chunk(2, dim=-1) # split along the width
        return self.b(torch.cat([right, left], dim=-1))


def test_split_and_concat_on_width():
    model = ChunkWidth()
    x = torch.randn(2, 3, 8, 8)
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=x)
    group = DG.get_pruning_group(model.a, tp.prune_conv_out_channels, idxs=[1, 3])
    b_idxs = [idxs for dep, idxs in group if dep.target.module is model.b]
    assert b_idxs and sorted(b_idxs[0]) == [1, 3], group
    group.prune()
    model(x)


if __name__ == "__main__":
    test_normalize_saved_dim()
    test_rope_concat_on_head_dim()
    test_rope_attention_head_pruning()
    test_concat_last_dim_2d()
    test_split_and_concat_on_width()
