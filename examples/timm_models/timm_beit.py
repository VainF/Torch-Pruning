import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

import torch
import torch.nn as nn 
import timm
import torch_pruning as tp
from typing import Sequence

from timm.models.vision_transformer import Attention
import torch.nn.functional as F

def forward(self, x, shared_rel_pos_bias = None):
    B, N, C = x.shape

    qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim

    if self.fused_attn:
        rel_pos_bias = None
        if self.relative_position_bias_table is not None:
            rel_pos_bias = self._get_rel_pos_bias()
            if shared_rel_pos_bias is not None:
                rel_pos_bias = rel_pos_bias + shared_rel_pos_bias
        elif shared_rel_pos_bias is not None:
            rel_pos_bias = shared_rel_pos_bias

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=rel_pos_bias,
            dropout_p=self.attn_drop.p,
        )
    else:
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            attn = attn + self._get_rel_pos_bias()
        if shared_rel_pos_bias is not None:
            attn = attn + shared_rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, -1)
    #x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

# timm==0.9.2
# torch==1.12.1

timm_models = timm.list_models()
example_inputs = torch.randn(1,3,224,224)
imp = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")
prunable_list = []
unprunable_list = []
problem_with_input_shape = []

for i, model_name in enumerate(timm_models):
    if not model_name=='beit_base_patch16_224':
        continue
    
    print("Pruning %s..."%model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #if 'rexnet' in model_name or 'sequencer' in model_name or 'botnet' in model_name:  # pruning process stuck with that architectures - skip them.
    #    unprunable_list.append(model_name)
    #    continue
    try:
        model = timm.create_model(model_name, pretrained=False, no_jit=True).eval().to(device)
    except: # out of memory error
        model = timm.create_model(model_name, pretrained=False, no_jit=True).eval()
        device = 'cpu'
    ch_groups = {}
    for m in model.modules():
        if isinstance(m, timm.models.vision_transformer.Attention):
            m.forward = timm_attention_forward.__get__(m, Attention) # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
            ch_groups[m.qkv] = m.num_heads * 3

    input_size = model.default_cfg['input_size']
    example_inputs = torch.randn(1, *input_size).to(device)
    test_output = model(example_inputs)

    print(model)
    prunable = True
    #try:
    if True:
        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        pruner = tp.pruner.MagnitudePruner(
                        model, 
                        example_inputs, 
                        global_pruning=False, # If False, a uniform sparsity will be assigned to different layers.
                        importance=imp, # importance criterion for parameter selection
                        iterative_steps=1, # the number of iterations to achieve target sparsity
                        ch_sparsity=0.5,
                        ignored_layers=[model.head],
                        channel_groups=ch_groups,
                    )
        for g in pruner.step(interactive=True):
            #print(g)
            g.prune()

        # Modify the attention head size and all head size aftering pruning
        for m in model.modules():
            if isinstance(m, timm.models.vision_transformer.Attention):
                m.head_dim = m.qkv.out_features // (3 * m.num_heads)

        print(model)
        test_output = model(example_inputs)
        pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
        print("Base MACs: %d, Pruned MACs: %d"%(base_macs, pruned_macs))
        print("Base Params: %d, Pruned Params: %d"%(base_params, pruned_params))

    if prunable:
        prunable_list.append(model_name)
    else:
        unprunable_list.append(model_name)
    
    print("Prunable: %d models, \n %s\n"%(len(prunable_list), prunable_list))
    print("Unprunable: %d models, \n %s\n"%(len(unprunable_list), unprunable_list))