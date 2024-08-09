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

def main():
    timm_models = timm.list_models()
    if args.list_models:
        print(timm_models)
    if args.model is None: 
        return
    assert args.model in timm_models, "Model %s is not in timm model list: %s"%(args.model, timm_models)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = timm.create_model(args.model, pretrained=args.pretrained, no_jit=True).eval().to(device)

    imp = tp.importance.GroupNormImportance()
    print("Pruning %s..."%args.model)
        
    input_size = model.default_cfg['input_size']
    example_inputs = torch.randn(1, *input_size).to(device)
    test_output = model(example_inputs)
    ignored_layers = []
    num_heads = {}
    pruning_ratio_dict = {}
    import random

    population = [
        [0.265625,0.234375,0.265625,0.265625,0.93359375,0.328125,0.2265625,0.58984375,0.54296875,0.701171875,0.919921875,0.04296875,0.796875,0.240966796875,0.07763671875],
        [0.96875,0.578125,0.3515625,0.6328125,0.7578125,0.7109375,0.8984375,0.533203125,0.0703125,0.697265625,0.451171875,0.626953125,0.935546875,0.294921875,0.5244140625],
        [0.25,0.421875,0.171875,0.4921875,0.71875,0.51953125,0.71875,0.876953125,0.896484375,0.626953125,0.646484375,0.490234375,0.65234375,0.599609375,0.0341796875],
        [0.015625,0.015625,0.078125,0.4375,0.59375,0.6953125,0.73828125,0.611328125,0.787109375,0.76171875,0.25,0.427734375,0.154296875,0.592529296875,0.298583984375]
    ]

    for ratios in population:
        k = 0
        for m in model.modules():
            #if hasattr(m, 'head'): #isinstance(m, nn.Linear) and m.out_features == model.num_classes:
            if isinstance(m, nn.Linear) and m.out_features == model.num_classes:
                ignored_layers.append(m)
                print("Ignore classifier layer: ", m)
        
            # Attention layers
            if hasattr(m, 'num_heads'):
                if hasattr(m, 'qkv'):
                    num_heads[m.qkv] = m.num_heads
                    print("Attention layer: ", m.qkv, m.num_heads)
                elif hasattr(m, 'qkv_proj'):
                    num_heads[m.qkv_proj] = m.num_heads
            
            elif isinstance(m, nn.Conv2d):
                pruning_ratio_dict[m] = ratios[k]
                k+=1


        print("========Before pruning========")
        print(model)
        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        pruner = tp.pruner.MetaPruner(
                        model, 
                        example_inputs, 
                        global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
                        importance=imp, # importance criterion for parameter selection
                        iterative_steps=1, # the number of iterations to achieve target pruning ratio
                        pruning_ratio=args.pruning_ratio, # target pruning ratio
                        pruning_ratio_dict=pruning_ratio_dict,
                        num_heads=num_heads,
                        ignored_layers=ignored_layers,
                    )
        for g in pruner.step(interactive=True):
            g.prune()

        for m in model.modules():
            # Attention layers
            if hasattr(m, 'num_heads'):
                if hasattr(m, 'qkv'):
                    m.num_heads = num_heads[m.qkv]
                    m.head_dim = m.qkv.out_features // (3 * m.num_heads)
                elif hasattr(m, 'qkv_proj'):
                    m.num_heads = num_heads[m.qqkv_projkv]
                    m.head_dim = m.qkv_proj.out_features // (3 * m.num_heads)

        print("========After pruning========")
        print(model)
        test_output = model(example_inputs)
        pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
        print("MACs: %.4f G => %.4f G"%(base_macs/1e9, pruned_macs/1e9))
        print("Params: %.4f M => %.4f M"%(base_params/1e6, pruned_params/1e6))

if __name__=='__main__':
    main()