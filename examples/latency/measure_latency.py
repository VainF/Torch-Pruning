import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import torch.nn.functional as F
import torch_pruning as tp
import timm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Latency Measurement')
    parser.add_argument('--model_name', default='vit_base_patch16_224', type=str, help='model name in timm')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for latency measurement')
    parser.add_argument('--pruning_ratio', default=0.5, type=float, help='prune ratio')
    parser.add_argument('--bottleneck', default=False, action='store_true', help='bottleneck or uniform')
    parser.add_argument('--pruning_type', default='l1', type=str, help='pruning type', choices=['random', 'l2', 'l1'])
    parser.add_argument('--global_pruning', default=False, action='store_true', help='global pruning')
    parser.add_argument('--prune_num_heads', default=False, action='store_true', help='global pruning')
    parser.add_argument('--head_pruning_ratio', default=0.0, type=float, help='head pruning ratio')
    parser.add_argument('--use_imagenet_mean_std', default=False, action='store_true', help='use imagenet mean and std')
    parser.add_argument('--save_as', default=None, type=str, help='save the pruned model')
    args = parser.parse_args()
    return args

# Here we re-implement the forward function of timm.models.vision_transformer.Attention
# as the original forward function requires the input and output channels to be identical.
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
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    example_inputs = torch.randn(1,3,224,224)
    imp = tp.importance.RandomImportance()
    # Load the model
    model = timm.create_model(args.model_name, pretrained=True).eval().to(device)

    print("Pruning %s..."%args.model_name)
    input_size = [3, 224, 224]
    example_inputs = torch.randn(1, *input_size, dtype=torch.float32).to(device)
    benchmark_inputs = torch.randn(args.batch_size, *input_size, dtype=torch.float32).to(device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    base_latency, _ = tp.utils.benchmark.measure_latency(model, benchmark_inputs, 20, 5)
    print("Base Latency: {} ms, Base MACs: {} G, Base Params: {} M".format(base_latency, base_macs/1e9, base_params/1e6))
    num_heads = {}

    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
            ignored_layer_outputs = [layer]

    for m in model.modules():
        if isinstance(m, timm.models.vision_transformer.Attention):
            m.forward = forward.__get__(m, timm.models.vision_transformer.Attention) # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
            num_heads[m.qkv] = m.num_heads 
        if args.bottleneck and isinstance(m, timm.models.vision_transformer.Mlp): 
            ignored_layer_outputs.append(m.fc2) # only prune the internal layers of FFN & Attention
    
    pruner = tp.pruner.MetaPruner(
        model, 
        example_inputs, 
        iterative_steps=10,
        global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
        importance=imp, # importance criterion for parameter selection
        pruning_ratio=1, # target pruning ratio
        ignored_layer_outputs=ignored_layer_outputs,
        num_heads=num_heads, # number of heads in self attention
        prune_num_heads=args.prune_num_heads, # reduce num_heads by pruning entire heads (default: False)
        prune_head_dims=not args.prune_num_heads, # reduce head_dim by pruning featrues dims of each head (default: True)
        head_pruning_ratio=0.5, #args.head_pruning_ratio, # remove 50% heads, only works when prune_num_heads=True (default: 0.0)
        round_to=8 # round_to=8 for fp16 and round_to=4 for tf32 https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
    )     

    for it in range(10):
        for i, g in enumerate(pruner.step(interactive=True)):
            g.prune()
        for m in model.modules():
            if isinstance(m, timm.models.vision_transformer.Attention):
                m.num_heads = pruner.num_heads[m.qkv]
                m.head_dim = m.qkv.out_features // (3 * m.num_heads)
        pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
        with torch.no_grad():
            pruned_latency, _ = tp.utils.benchmark.measure_latency(model, benchmark_inputs, 20, 5)
            print("Iter {}, Pruned Latency: {:.2f} ms, Pruned MACs: {:.2f} G, Pruned Params: {:.2f} M".format(it, pruned_latency, pruned_macs/1e9, pruned_params/1e6))

if __name__=='__main__':
    main()