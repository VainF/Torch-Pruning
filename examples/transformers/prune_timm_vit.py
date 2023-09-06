import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

import torch
import torch.nn as nn 
import timm
import torch_pruning as tp
from typing import Sequence

from timm.models.vision_transformer import Attention
import torch.nn.functional as F

# Here we re-implement the forward function of timm.models.vision_transformer.Attention
# as the original forward function requires the input and output channels to be identical.
def timm_attention_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    #x = x.transpose(1, 2).reshape(B, N, C) # this line forces the input and output channels to be identical.
    x = x.transpose(0, 1).reshape(B, N, -1) 
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

# timm==0.9.2
# torch==1.12.1
example_inputs = torch.randn(1,3,224,224)
imp = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")
prunable_list = []
unprunable_list = []
problem_with_input_shape = []
model_name = 'vit_base_patch16_224'
print("Pruning %s..."%model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    model = timm.create_model(model_name, pretrained=False, no_jit=True).eval().to(device)
except: # out of memory error
    model = timm.create_model(model_name, pretrained=False, no_jit=True).eval()
    device = 'cpu'
ch_groups = {}
ignored_layers = [model.head]
for m in model.modules():
    if isinstance(m, timm.models.vision_transformer.Attention):
        m.forward = timm_attention_forward.__get__(m, Attention) # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
        ch_groups[m.qkv] = m.num_heads * 3
    if isinstance(m, timm.models.vision_transformer.Mlp):
        ignored_layers.append(m.fc2)

input_size = model.default_cfg['input_size']
example_inputs = torch.randn(1, *input_size).to(device)
test_output = model(example_inputs)

print(model)
base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
pruner = tp.pruner.MagnitudePruner(
                model, 
                example_inputs, 
                global_pruning=False, # If False, a uniform sparsity will be assigned to different layers.
                importance=imp, # importance criterion for parameter selection
                iterative_steps=1, # the number of iterations to achieve target sparsity
                ch_sparsity=0.75,
                ignored_layers=ignored_layers,
                channel_groups=ch_groups,
            )
for g in pruner.step(interactive=True):
    g.prune()

# Modify the attention head size and all head size aftering pruning
for m in model.modules():
    if isinstance(m, timm.models.vision_transformer.Attention):
        m.head_dim = m.qkv.out_features // (3 * m.num_heads)

print(model)
test_output = model(example_inputs)
pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
print("Base MACs: %.2f G, Pruned MACs: %.2f G"%(base_macs/1e9, pruned_macs/1e9))
print("Base Params: %.2f M, Pruned Params: %.2f G"%(base_params/1e6, pruned_params/1e6))
