# Performance Benchmarks (Beta version)

## 1. ResNet-56 / CIFAR-10 / 2.00x - 2.55x

| Method | Base (%) | Pruned (%) | $\Delta$ Acc (%) | Speed Up |
|:--    |:--:  |:--:    |:--: |:--:      |
| NIPS [[1]](#1)  | -    | -      |-0.03 | 1.76x    |
| Geometric [[2]](#2) | 93.59 | 93.26 | -0.33 | 1.70x |
| Polar [[3]](#3)  | 93.80 | 93.83 | +0.03 |1.88x |
| CP  [[4]](#4)   | 92.80 | 91.80 | -1.00 |2.00x |
| AMC [[5]](#5)   | 92.80 | 91.90 | -0.90 |2.00x |
| HRank [[6]](#6) | 93.26 | 92.17 | -0.09 |2.00x |
| SFP  [[7]](#7)  | 93.59 | 93.36 | -0.23 |2.11x |
| ResRep [[8]](#8) | 93.71 | 93.71 | +0.00 |2.12x |
| Group-L1 | 93.53 | 92.93 | -0.60 | 2.12x |
| Group-BN | 93.53 | 93.29 | -0.24 | 2.12x |
| Group-GReg | 93.53 | 93.55 | +0.02 | 2.12x |
| Ours w/o SL | 93.53 | 93.46 | -0.07 | 2.11x |
| **Ours** | 93.53 | **93.77** | +0.38 | 2.13x |
||
| GBN [[9]](#9) | 93.10 |  92.77 | -0.33 | 2.51x |
| AFP [[10]](#10)  | 93.93 | 92.94 | -0.99 | 2.56x |
| C-SGD [[11]](#11) | 93.39 | 93.44 | +0.05 | 2.55x |
| GReg-1 [[12]](#12)  | 93.36 | 93.18 | -0.18 | 2.55x |
| GReg-2 [[12]](#12)  | 93.36 | 93.36 | -0.00 | 2.55x |
| Ours w/o SL | 93.53 | 93.36 | -0.17 | 2.51x |
| **Ours** | 93.53 | **93.64** | +0.11 | 2.57x |

**Note 1:** $\text{speed up} = \frac{\text{Base MACs}}{\text{Pruned MACs}}$

**Note 2:** Baseline methods are not implemented in this repo, because they may require additional modifications to the standard models and training scripts. We are working to support more algorithms.

**Note 3:** Donwload pretrained resnet-56 from [Dropbox](https://www.dropbox.com/sh/71s2rlt5zr83i4v/AAAjBCwslVf89TjJ49NHl0Epa?dl=0) or [Github Release](https://github.com/VainF/Torch-Pruning/releases/tag/v1.1.4)

**Note 4:** Training logs are available at [run/](https://github.com/VainF/Torch-Pruning/tree/master/benchmarks/run).

**Note 5:** "w/o SL" = "without sparse learning"


### 1.1 Download pre-trained models for reproducibility
```bash
wget https://github.com/VainF/Torch-Pruning/releases/download/v1.1.4/cifar10_resnet56.pth
```
or train a new model:
```python
python main.py --mode pretrain --dataset cifar10 --model resnet56 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180 
```

### 1.2 CIFAR-10 Pruning

#### - L1-Norm Pruner (Group-L1)
A group-level pruner adapted from [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
```bash
# 2.11x
python main.py --mode prune --model resnet56 --batch-size 128 --restore </path/to/pretrained/model> --dataset cifar10  --method l1 --speed-up 2.11 --global-pruning
```

#### - BN Pruner (Group-BN)
A group-level pruner adapted from [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)
```bash
# 2.11x
python main.py --mode prune --model resnet56 --batch-size 128 --restore </path/to/pretrained/model> --dataset cifar10  --method slim --speed-up 2.11 --global-pruning --reg 1e-5
```

#### - Growing Regularization (Group-GReg)
A group-level pruner adapted from [Neural Pruning via Growing Regularization](https://arxiv.org/abs/2012.09243)
```bash
# 2.11x
python main.py --mode prune --model resnet56 --batch-size 128 --restore </path/to/pretrained/model> --dataset cifar10  --method growing_reg --speed-up 2.11 --global-pruning --reg 1e-4 --delta_reg 1e-5
```

#### - Group Pruner (This Work)
```bash
# 2.11x without sparse learning (Ours w/o SL)
python main.py --mode prune --model resnet56 --batch-size 128 --restore </path/to/pretrained/model> --dataset cifar10  --method group_norm --speed-up 2.11 --global-pruning

# 2.55x without sparse learning (Ours w/o SL)
python main.py --mode prune --model resnet56 --batch-size 128 --restore </path/to/pretrained/model> --dataset cifar10  --method group_norm --speed-up 2.55 --global-pruning

# 2.11x (Ours)
python main.py --mode prune --model resnet56 --batch-size 128 --restore </path/to/pretrained/model> --dataset cifar10  --method group_sl --speed-up 2.11 --global-pruning --reg 5e-4

# 2.55x (Ours)
python main.py --mode prune --model resnet56 --batch-size 128 --restore </path/to/pretrained/model> --dataset cifar10  --method group_sl --speed-up 2.55 --global-pruning --reg 5e-4
```

## 2. VGG-19 / CIFAR-100 / 8.8x

| Method | Base (%) | Pruned (%) | $\Delta$ Acc (%) | Speed Up |
|:--    |:--:  |:--:    |:--: |:--:      |
| OBD [[13]](#13) | 73.34 | 60.70 | -12.64 | 5.73x |
| OBD [[13]](#13) | 73.34 | 60.66 | -12.68 | 6.09x |
| EigenD [[13]](#13) | 73.34 | 65.18 | -8.16 |  8.80× |
| GReg-1 [[12]](#12) | 74.02 | 67.55 | -6.67 | 8.84× |
| GReg-2 [[12]](#12) | 74.02 | 67.75 | -6.27 | 8.84× |
| Ours w/o SL | 73.50 | 67.60 | -5.44 |  8.87x |
| Ours | 73.50 | 70.39  | -3.11 | 8.92× |

### 2.1 Download pre-trained models for reproducibility
```bash
wget https://github.com/VainF/Torch-Pruning/releases/download/v1.1.4/cifar100_vgg19.pth
```
or train a new model:
```python
python main.py --mode pretrain --dataset cifar100 --model vgg19 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180 
```

### 2.2 CIFAR-100 Pruning

#### - Group Pruner (This Work)
```bash
# 8.84x without sparse learning (Ours w/o SL)
python main.py --mode prune --model vgg19 --batch-size 128 --restore </path/to/pretrained/model> --dataset cifar100  --method group_norm --speed-up 8.84 --global-pruning

# 8.84x (Ours)
python main.py --mode prune --model vgg19 --batch-size 128 --restore </path/to/pretrained/model> --dataset cifar100  --method group_sl --speed-up 8.84 --global-pruning --reg 5e-4
```

## 3. ResNet50 / ImageNet / 2.00 GMACs

#### - Group L1-Norm Pruner (without Sparse Learning)
```python
python -m torch.distributed.launch --nproc_per_node=4 --master_port 18119 --use_env main_imagenet.py --model resnet50 --epochs 90 --batch-size 64 --lr-step-size 30 --lr 0.01 --prune --method l1 --pretrained --output-dir run/imagenet/resnet50_sl --target-flops 2.00 --cache-dataset --print-freq 100 --workers 16 --data-path PATH_TO_IMAGENET --output-dir PATH_TO_OUTPUT_DIR # &> output.log
```

For the latest results on Vision Transformers, please refer to [examples/transformers](https://github.com/VainF/Torch-Pruning/tree/master/examples/transformers).


**More results will be released soon!**

## 4. Latency Test (ResNet-50, Batch Size 64)
```bash
python benchmark_latency.py
```
```
[Iter 0]        Pruning ratio: 0.00,         MACs: 4.12 G,   Params: 25.56 M,        Latency: 45.22 ms +- 0.03 ms
[Iter 1]        Pruning ratio: 0.05,         MACs: 3.68 G,   Params: 22.97 M,        Latency: 46.53 ms +- 0.06 ms
[Iter 2]        Pruning ratio: 0.10,         MACs: 3.31 G,   Params: 20.63 M,        Latency: 43.85 ms +- 0.08 ms
[Iter 3]        Pruning ratio: 0.15,         MACs: 2.97 G,   Params: 18.36 M,        Latency: 41.22 ms +- 0.10 ms
[Iter 4]        Pruning ratio: 0.20,         MACs: 2.63 G,   Params: 16.27 M,        Latency: 39.28 ms +- 0.20 ms
[Iter 5]        Pruning ratio: 0.25,         MACs: 2.35 G,   Params: 14.39 M,        Latency: 34.60 ms +- 0.19 ms
[Iter 6]        Pruning ratio: 0.30,         MACs: 2.02 G,   Params: 12.46 M,        Latency: 33.38 ms +- 0.27 ms
[Iter 7]        Pruning ratio: 0.35,         MACs: 1.74 G,   Params: 10.75 M,        Latency: 31.46 ms +- 0.20 ms
[Iter 8]        Pruning ratio: 0.40,         MACs: 1.50 G,   Params: 9.14 M,         Latency: 29.04 ms +- 0.19 ms
[Iter 9]        Pruning ratio: 0.45,         MACs: 1.26 G,   Params: 7.68 M,         Latency: 27.47 ms +- 0.28 ms
[Iter 10]       Pruning ratio: 0.50,         MACs: 1.07 G,   Params: 6.41 M,         Latency: 20.68 ms +- 0.13 ms
[Iter 11]       Pruning ratio: 0.55,         MACs: 0.85 G,   Params: 5.14 M,         Latency: 20.48 ms +- 0.21 ms
[Iter 12]       Pruning ratio: 0.60,         MACs: 0.67 G,   Params: 4.07 M,         Latency: 18.12 ms +- 0.15 ms
[Iter 13]       Pruning ratio: 0.65,         MACs: 0.53 G,   Params: 3.10 M,         Latency: 15.19 ms +- 0.01 ms
[Iter 14]       Pruning ratio: 0.70,         MACs: 0.39 G,   Params: 2.28 M,         Latency: 13.47 ms +- 0.01 ms
[Iter 15]       Pruning ratio: 0.75,         MACs: 0.29 G,   Params: 1.61 M,         Latency: 10.07 ms +- 0.01 ms
[Iter 16]       Pruning ratio: 0.80,         MACs: 0.18 G,   Params: 1.01 M,         Latency: 8.96 ms +- 0.02 ms
[Iter 17]       Pruning ratio: 0.85,         MACs: 0.10 G,   Params: 0.57 M,         Latency: 7.03 ms +- 0.04 ms
[Iter 18]       Pruning ratio: 0.90,         MACs: 0.05 G,   Params: 0.25 M,         Latency: 5.81 ms +- 0.03 ms
[Iter 19]       Pruning ratio: 0.95,         MACs: 0.01 G,   Params: 0.06 M,         Latency: 5.70 ms +- 0.03 ms
[Iter 20]       Pruning ratio: 1.00,         MACs: 0.01 G,   Params: 0.06 M,         Latency: 5.71 ms +- 0.03 ms
```

## Benchmark of Importance Criteria

ResNet50 pre-trained on ImageNet-1K, local pruning without fine-tuning.

```bash
python benchmark_importance_criteria.py
```

``Single-layer`` means ``group_reduction='first'``, which only leverages the first layer of a group for importance estimation.

<div align="center">
<img src="https://github.com/VainF/Torch-Pruning/assets/18592211/775eb01a-4610-4637-90bd-ff53f7ea2d31" width="45%"></img>
<img src="https://github.com/VainF/Torch-Pruning/assets/18592211/085aa9ec-a520-4939-97f4-46f65b124929" width="45%"></img>
</div>

## References

<a id="1">[1]</a> Nisp: Pruning networks using neuron importance score propagation. 

<a id="2">[2]</a> Filter pruning via geometric median for deep convolutional neural networks acceleration. 

<a id="3">[3]</a> Neuron-level structured pruning using polarization regularizer.  

<a id="4">[4]</a> Pruning Filters for Efficient ConvNets.

<a id="5">[5]</a> Amc: Automl for model compression and acceleration on mobile devices.

<a id="6">[6]</a> Hrank: Filter pruning using high-rank feature map.

<a id="7">[7]</a> Soft filter pruning for accelerating deep convolutional neural networks

<a id="8">[8]</a> Resrep: Lossless cnn pruning via decoupling remembering and forgetting.

<a id="9">[9]</a> Gate decorator: Global filter pruning method for accelerating deep convolutional neural networks.

<a id="10">[10]</a> Auto-balanced filter pruning for efficient convolutional neural networks.

<a id="11">[11]</a> Centripetal sgd for pruning very deep convolutional networks with complicated structure

<a id="12">[12]</a> Neural pruning via growing regularization

<a id="13">[13]</a>  Eigendamage: Structured pruning in the kroneckerfactored eigenbasis.
