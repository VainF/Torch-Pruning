import sys, os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch_pruning as tp
import torch
import timm
import torch.nn.functional as F

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vit_base_patch16_224', help='model name or path')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
args = parser.parse_args()

def forward(self, x):
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

    x = x.transpose(1, 2).reshape(B, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.isfile(args.model):
        loaded_pth = torch.load(args.model, map_location='cpu')
        if isinstance(loaded_pth, dict):
            model = loaded_pth['model'].to(device)
        else:
            model = loaded_pth.to(device)
    else:
        model = timm.create_model(args.model, pretrained=True).to(device)
        
    for m in model.modules():
        if isinstance(m, timm.models.vision_transformer.Attention):
            m.forward = forward.__get__(m, timm.models.vision_transformer.Attention) # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module

    example_input = torch.rand(args.batch_size, 3, 224, 224).to(device)
    macs, params = tp.utils.count_ops_and_params(model, example_input)
    latency_mu, latency_std = estimate_latency(model, example_input)
    print(f"MACs: {macs/1e9:.2f} G, \tParams: {params/1e6:.2f} M, \tLatency: {latency_mu:.2f} ms +- {latency_std:.2f} ms")

def estimate_latency(model, example_inputs, repetitions=300):
    import numpy as np
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings=np.zeros((repetitions,1))

    for _ in range(50):
        _ = model(example_inputs)

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(example_inputs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn

if __name__=='__main__':
    main()