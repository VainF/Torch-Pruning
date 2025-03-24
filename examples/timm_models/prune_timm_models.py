import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
os.environ['TIMM_FUSED_ATTN'] = '0'
import torch
import torch.nn as nn 
import torch.nn.functional as F
from typing import Sequence
import timm
from timm.models.vision_transformer import Attention
import torch_pruning as tp
import argparse

parser = argparse.ArgumentParser(description='Prune timm models')
parser.add_argument('--model', default=None, type=str, help='model name')
parser.add_argument('--pruning_ratio', default=0.5, type=float, help='channel pruning ratio')
parser.add_argument('--global_pruning', default=False, action='store_true', help='global pruning')
parser.add_argument('--pretrained', default=False, action='store_true', help='global pruning')
parser.add_argument('--list_models', default=False, action='store_true', help='list all models in timm')
args = parser.parse_args()


def forward(self, x):
    """https://github.com/huggingface/pytorch-image-models/blob/054c763fcaa7d241564439ae05fbe919ed85e614/timm/models/vision_transformer.py#L79"""
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

    x = x.transpose(1, 2).reshape(B, N, -1) # original implementation: x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def main():
    timm_models = timm.list_models()
    if args.list_models:
        print(timm_models)
    if args.model is None: 
        return
    assert args.model in timm_models, "Model %s is not in timm model list: %s"%(args.model, timm_models)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = timm.create_model(args.model, pretrained=args.pretrained, no_jit=True).eval().to(device)
    
    # Set up the pruner (importance score, num heads, etc.)
    imp = tp.importance.GroupMagnitudeImportance()
    input_size = model.default_cfg['input_size']
    example_inputs = torch.randn(1, *input_size).to(device)
    test_output = model(example_inputs)
    # In practice, you should add the ouput layer to ignored_layers.
    ignored_layers = [] 
    if hasattr(model, 'head'):
        if isinstance(model.head, nn.Linear):
            ignored_layers.append(model.head)
        else:
            last_linear = None
            for m in model.head.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None:
                ignored_layers.append(last_linear)
    else:
        import warnings
        warnings.warn("Cannot find the output layer in the model. Please add the output layer to ignored_layers.")

    num_heads = {}
    pruning_ratio_dict = {}
    for m in model.modules():
        if isinstance(m, timm.models.vision_transformer.Attention):
            num_heads[m.qkv] = m.num_heads 
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    pruner = tp.pruner.BasePruner(
                    model, 
                    example_inputs, 
                    global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
                    importance=imp, # importance criterion for parameter selection
                    iterative_steps=1, # the number of iterations to achieve target pruning ratio
                    pruning_ratio=args.pruning_ratio, # target pruning ratio
                    pruning_ratio_dict=pruning_ratio_dict,
                    num_heads=num_heads,
                    ignored_layers=ignored_layers,
                    round_to=8,
                )

    print("Pruning %s..."%args.model)
    tp.utils.print_tool.before_pruning(model)
    for g in pruner.step(interactive=True):
        g.prune()

    for m in model.modules():
        # Attention layers
        if hasattr(m, 'num_heads'):
            if hasattr(m, 'qkv'):
                m.forward = forward.__get__(m, timm.models.vision_transformer.Attention)
                m.num_heads = num_heads[m.qkv]
                m.head_dim = m.qkv.out_features // (3 * m.num_heads)
            elif hasattr(m, 'qkv_proj'):
                m.num_heads = num_heads[m.qqkv_projkv]
                m.head_dim = m.qkv_proj.out_features // (3 * m.num_heads)
    tp.utils.print_tool.after_pruning(model, do_print=True)
    test_output = model(example_inputs)
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    print("MACs: %.4f G => %.4f G"%(base_macs/1e9, pruned_macs/1e9))
    print("Params: %.4f M => %.4f M"%(base_params/1e6, pruned_params/1e6))

if __name__=='__main__':
    main()