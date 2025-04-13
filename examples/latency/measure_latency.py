import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import torch.nn.functional as F
import torch_pruning as tp
import timm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Timm ViT Pruning')
    parser.add_argument('--model_name', default='vit_base_patch16_224', type=str, help='model name')
    parser.add_argument('--data_path', default='data/imagenet', type=str, help='model name')
    parser.add_argument('--taylor_batchs', default=10, type=int, help='number of batchs for taylor criterion')
    parser.add_argument('--pruning_ratio', default=0.5, type=float, help='prune ratio')
    parser.add_argument('--bottleneck', default=False, action='store_true', help='bottleneck or uniform')
    parser.add_argument('--pruning_type', default='l1', type=str, help='pruning type', choices=['random', 'taylor', 'l2', 'l1', 'hessian'])
    parser.add_argument('--test_accuracy', default=False, action='store_true', help='test accuracy')
    parser.add_argument('--global_pruning', default=False, action='store_true', help='global pruning')
    parser.add_argument('--prune_num_heads', default=False, action='store_true', help='global pruning')
    parser.add_argument('--head_pruning_ratio', default=0.0, type=float, help='head pruning ratio')
    parser.add_argument('--use_imagenet_mean_std', default=False, action='store_true', help='use imagenet mean and std')
    parser.add_argument('--train_batch_size', default=64, type=int, help='train batch size')
    parser.add_argument('--val_batch_size', default=128, type=int, help='val batch size')
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
    test_inputs = torch.randn(128, *input_size, dtype=torch.float32).to(device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    base_latency, _ = tp.utils.benchmark.measure_latency(model, test_inputs, 20, 5)
    print("Base Latency: {} ms, Base MACs: {}, Base Params: {}".format(base_latency, base_macs, base_params))
    num_heads = {}

    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
            ignored_layers = [layer]

    for m in model.modules():
        if isinstance(m, timm.models.vision_transformer.Attention):
            m.forward = forward.__get__(m, timm.models.vision_transformer.Attention) # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
            num_heads[m.qkv] = m.num_heads 
        if args.bottleneck and isinstance(m, timm.models.vision_transformer.Mlp): 
            ignored_layers.append(m.fc2) # only prune the internal layers of FFN & Attention
    pruner = tp.pruner.BasePruner(
        model, 
        example_inputs, 
        iterative_steps=10,
        global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
        importance=imp, # importance criterion for parameter selection
        pruning_ratio=1, # target pruning ratio
        ignored_layers=ignored_layers,
        num_heads=num_heads, # number of heads in self attention
        prune_num_heads=args.prune_num_heads, # reduce num_heads by pruning entire heads (default: False)
        prune_head_dims=not args.prune_num_heads, # reduce head_dim by pruning featrues dims of each head (default: True)
        head_pruning_ratio=0.5, #args.head_pruning_ratio, # remove 50% heads, only works when prune_num_heads=True (default: 0.0)
        round_to=8 # round_to=8 for fp16 and round_to=4 for tf32 https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
    )      

    for it in range(10):
        for i, g in enumerate(pruner.step(interactive=True)):
            g.prune()

        head_id = 0
        for m in model.modules():
            if isinstance(m, timm.models.vision_transformer.Attention):
                m.num_heads = pruner.num_heads[m.qkv]
                m.head_dim = m.qkv.out_features // (3 * m.num_heads)
        pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
        pruned_latency, _ = tp.utils.benchmark.measure_latency(model, test_inputs, 20, 5)
        print("Iter {}, Pruned Latency: {:.2f} ms, Pruned MACs: {:.2f} G, Pruned Params: {:.2f} M".format(it, pruned_latency, pruned_macs/1e9, pruned_params/1e6))

# Iter 0, Pruned Latency: 248.28 ms, Pruned MACs: 14.35 G, Pruned Params: 70.29 M
# Iter 1, Pruned Latency: 189.74 ms, Pruned MACs: 11.47 G, Pruned Params: 55.89 M
# Iter 2, Pruned Latency: 147.47 ms, Pruned MACs: 8.71 G, Pruned Params: 42.38 M
# Iter 3, Pruned Latency: 123.83 ms, Pruned MACs: 6.50 G, Pruned Params: 31.38 M
# Iter 4, Pruned Latency: 77.39 ms, Pruned MACs: 4.62 G, Pruned Params: 22.05 M
# Iter 5, Pruned Latency: 76.71 ms, Pruned MACs: 3.04 G, Pruned Params: 14.25 M
# Iter 6, Pruned Latency: 44.16 ms, Pruned MACs: 1.80 G, Pruned Params: 8.22 M
# Iter 7, Pruned Latency: 27.29 ms, Pruned MACs: 0.81 G, Pruned Params: 3.61 M
# Iter 8, Pruned Latency: 23.87 ms, Pruned MACs: 0.25 G, Pruned Params: 0.98 M
# Iter 9, Pruned Latency: 23.82 ms, Pruned MACs: 0.25 G, Pruned Params: 0.98 M   
        
# Iter 0, Pruned Latency: 260.67 ms, Pruned MACs: 14.35 G, Pruned Params: 70.29 M
# Iter 1, Pruned Latency: 193.27 ms, Pruned MACs: 11.47 G, Pruned Params: 55.89 M
# Iter 2, Pruned Latency: 152.21 ms, Pruned MACs: 8.71 G, Pruned Params: 42.38 M
# Iter 3, Pruned Latency: 127.89 ms, Pruned MACs: 6.50 G, Pruned Params: 31.38 M
# Iter 4, Pruned Latency: 80.10 ms, Pruned MACs: 4.62 G, Pruned Params: 22.05 M
# Iter 5, Pruned Latency: 79.03 ms, Pruned MACs: 3.04 G, Pruned Params: 14.25 M
# Iter 6, Pruned Latency: 45.70 ms, Pruned MACs: 1.80 G, Pruned Params: 8.22 M
# Iter 7, Pruned Latency: 28.08 ms, Pruned MACs: 0.81 G, Pruned Params: 3.61 M
# Iter 8, Pruned Latency: 24.29 ms, Pruned MACs: 0.25 G, Pruned Params: 0.98 M
# Iter 9, Pruned Latency: 24.38 ms, Pruned MACs: 0.25 G, Pruned Params: 0.98 M
if __name__=='__main__':
    main()