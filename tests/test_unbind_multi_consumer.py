import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, h):
        dtype = h.dtype
        h = h.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.eps)
        return (self.weight * h).to(dtype)


class QKNormAttention(nn.Module):
    """timm-style attention with qk-norm: q and k are each consumed by two ops (pow and mul) of the RMSNorm."""
    def __init__(self, dim=96, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.q_norm, self.k_norm = RMSNorm(dim // num_heads), RMSNorm(dim // num_heads)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, -1))


class Net(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.embed = nn.Linear(16, dim)
        self.attn = QKNormAttention(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.embed(x)
        x = x + self.attn(x)
        return self.head(x.mean(1))


def test_unbind_chunks():
    # https://github.com/VainF/Torch-Pruning/issues/524
    # the unbind has 3 outputs, not one per downstream op (5 here)
    model = Net()
    x = torch.randn(2, 5, 16)
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=x)
    DG.get_pruning_group(model.attn.qkv, tp.prune_linear_out_channels, idxs=list(range(288)))
    unbind = [n for n in DG.module2node.values() if n.type == tp.ops.OPTYPE.UNBIND]
    assert len(unbind) == 1
    assert unbind[0].module.offsets == [0, 96, 192, 288]
    # feature 0 of v is coupled with feature 0 of q and k through the attention
    group = DG.get_pruning_group(model.attn.proj, tp.prune_linear_in_channels, idxs=[0])
    qkv_idxs = [idxs for dep, idxs in group if dep.target.module is model.attn.qkv]
    assert sorted(qkv_idxs[0]) == [0, 96, 192], group


def test_unbind_pruning():
    model = Net()
    x = torch.randn(2, 5, 16)
    pruner = tp.pruner.MetaPruner(model, x, importance=tp.importance.GroupMagnitudeImportance(), pruning_ratio=0.25,
                                  num_heads={model.attn.qkv: 4}, ignored_layers=[model.head])
    pruner.step()
    head_dim = model.attn.qkv.out_features // (3 * model.attn.num_heads)
    assert head_dim == 18
    assert model.attn.q_norm.weight.shape[0] == model.attn.k_norm.weight.shape[0] == head_dim
    model(x)


def test_unbind_head_pruning():
    # removing whole heads keeps the (head_dim,) weights of the qk-norm
    model = Net()
    x = torch.randn(2, 5, 16)
    num_heads = {model.attn.qkv: 4}
    pruner = tp.pruner.MetaPruner(model, x, importance=tp.importance.GroupMagnitudeImportance(), pruning_ratio=0.5,
                                  num_heads=num_heads, prune_num_heads=True, prune_head_dims=False, head_pruning_ratio=0.5,
                                  ignored_layers=[model.head])
    pruner.step()
    model.attn.num_heads = num_heads[model.attn.qkv]
    assert model.attn.num_heads == 2 and model.attn.qkv.out_features == 3 * 2 * 24
    assert model.attn.q_norm.weight.shape[0] == model.attn.k_norm.weight.shape[0] == 24
    model(x)


if __name__ == "__main__":
    test_unbind_chunks()
    test_unbind_pruning()
    test_unbind_head_pruning()
