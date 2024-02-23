# Latency Test

## ViT

```bash
python3 measure_latency.py --model vit_base_patch16_224 --batch_size 32
```

Output:
```
Pruning vit_base_patch16_224...
Base Latency: 204.94448852539062 ms, Base MACs: 17.593519096 G, Base Params: 86.567656 M
Iter 0, Pruned Latency: 176.84 ms, Pruned MACs: 14.08 G, Pruned Params: 69.23 M
Iter 1, Pruned Latency: 141.99 ms, Pruned MACs: 10.98 G, Pruned Params: 53.94 M
Iter 2, Pruned Latency: 117.12 ms, Pruned MACs: 8.35 G, Pruned Params: 41.07 M
Iter 3, Pruned Latency: 89.43 ms, Pruned MACs: 6.00 G, Pruned Params: 29.51 M
Iter 4, Pruned Latency: 70.27 ms, Pruned MACs: 4.62 G, Pruned Params: 22.05 M
Iter 5, Pruned Latency: 55.78 ms, Pruned MACs: 2.91 G, Pruned Params: 13.78 M
Iter 6, Pruned Latency: 41.86 ms, Pruned MACs: 1.59 G, Pruned Params: 7.49 M
Iter 7, Pruned Latency: 29.14 ms, Pruned MACs: 0.68 G, Pruned Params: 3.24 M
Iter 8, Pruned Latency: 24.12 ms, Pruned MACs: 0.28 G, Pruned Params: 1.01 M
Iter 9, Pruned Latency: 24.09 ms, Pruned MACs: 0.28 G, Pruned Params: 1.01 M
```

## ResNet

```bash
python3 measure_latency.py --model resnet50 --batch_size 32
```

Output:
```
Pruning resnet50...
Base Latency: 51.954795837402344 ms, Base MACs: 4.121925096 G, Base Params: 25.557032 M
Iter 0, Pruned Latency: 46.91 ms, Pruned MACs: 3.22 G, Pruned Params: 20.42 M
Iter 1, Pruned Latency: 39.16 ms, Pruned MACs: 2.52 G, Pruned Params: 16.37 M
Iter 2, Pruned Latency: 35.75 ms, Pruned MACs: 1.94 G, Pruned Params: 12.64 M
Iter 3, Pruned Latency: 28.48 ms, Pruned MACs: 1.40 G, Pruned Params: 9.50 M
Iter 4, Pruned Latency: 23.46 ms, Pruned MACs: 1.07 G, Pruned Params: 6.92 M
Iter 5, Pruned Latency: 16.48 ms, Pruned MACs: 0.64 G, Pruned Params: 4.39 M
Iter 6, Pruned Latency: 11.95 ms, Pruned MACs: 0.35 G, Pruned Params: 2.62 M
Iter 7, Pruned Latency: 6.67 ms, Pruned MACs: 0.15 G, Pruned Params: 1.27 M
Iter 8, Pruned Latency: 5.20 ms, Pruned MACs: 0.05 G, Pruned Params: 0.41 M
Iter 9, Pruned Latency: 5.02 ms, Pruned MACs: 0.05 G, Pruned Params: 0.41 M
```

## ConvNext Base

```bash
python3 measure_latency.py --model convnext_base --batch_size 16
```

Output:
```
Pruning convnext_base...
Base Latency: 112.70329284667969 ms, Base MACs: 15.360289896 G, Base Params: 88.591464 M
Iter 0, Pruned Latency: 99.57 ms, Pruned MACs: 12.27 G, Pruned Params: 71.22 M
Iter 1, Pruned Latency: 84.07 ms, Pruned MACs: 9.73 G, Pruned Params: 56.58 M
Iter 2, Pruned Latency: 68.48 ms, Pruned MACs: 7.43 G, Pruned Params: 43.15 M
Iter 3, Pruned Latency: 56.65 ms, Pruned MACs: 5.48 G, Pruned Params: 31.94 M
Iter 4, Pruned Latency: 43.00 ms, Pruned MACs: 3.90 G, Pruned Params: 22.67 M
Iter 5, Pruned Latency: 33.70 ms, Pruned MACs: 2.43 G, Pruned Params: 14.35 M
Iter 6, Pruned Latency: 24.12 ms, Pruned MACs: 1.38 G, Pruned Params: 8.24 M
Iter 7, Pruned Latency: 18.81 ms, Pruned MACs: 0.61 G, Pruned Params: 3.67 M
Iter 8, Pruned Latency: 10.56 ms, Pruned MACs: 0.15 G, Pruned Params: 0.99 M
Iter 9, Pruned Latency: 10.41 ms, Pruned MACs: 0.15 G, Pruned Params: 0.99 M
```