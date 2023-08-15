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

```bash
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
python main.py --mode pretrain --dataset vgg19 --model resnet56 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180 
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

**More results will be released soon!**

## 4. Latency Test (ResNet-50, Batch Size 64)
```bash
python latency.py
```
```
Iter 0/20: MACs 11.580686863625 -> 11.580686863625, Params 60.192808 -> 60.192808, latency 103.74445037841797ms +- 0.1503811966662325ms
Iter 1/20: MACs 11.580686863625 -> 10.35356202084375, Params 60.192808 -> 53.975082, latency 109.17123962402344ms +- 0.1485938944854119ms
Iter 2/20: MACs 11.580686863625 -> 9.3215540500625, Params 60.192808 -> 48.5822, latency 102.61776382446288ms +- 0.14299805790749653ms
Iter 3/20: MACs 11.580686863625 -> 8.29743861528125, Params 60.192808 -> 43.110724, latency 93.73429733276367ms +- 0.15927156053040534ms
Iter 4/20: MACs 11.580686863625 -> 7.3762030005, Params 60.192808 -> 38.306114, latency 90.40859100341797ms +- 0.26119836770161675ms
Iter 5/20: MACs 11.580686863625 -> 6.54883777171875, Params 60.192808 -> 33.888798, latency 76.18217987060547ms +- 0.266677203754941ms
Iter 6/20: MACs 11.580686863625 -> 5.6343359209375, Params 60.192808 -> 29.265776, latency 76.30733261108398ms +- 0.05802990016940302ms
Iter 7/20: MACs 11.580686863625 -> 4.87963908215625, Params 60.192808 -> 25.332058, latency 70.61649337768554ms +- 0.06489935301185264ms
Iter 8/20: MACs 11.580686863625 -> 4.147461083375, Params 60.192808 -> 21.426626, latency 62.96350700378418ms +- 0.1294969014091071ms
Iter 9/20: MACs 11.580686863625 -> 3.50353660059375, Params 60.192808 -> 18.08118, latency 60.20911094665527ms +- 0.11310063116900947ms
Iter 10/20: MACs 11.580686863625 -> 2.9414402638125, Params 60.192808 -> 15.08866, latency 44.48813056945801ms +- 0.1265449951686286ms
Iter 11/20: MACs 11.580686863625 -> 2.339559565, Params 60.192808 -> 12.0585, latency 45.977784423828126ms +- 0.1454013610733702ms
Iter 12/20: MACs 11.580686863625 -> 1.86217569825, Params 60.192808 -> 9.585788, latency 40.39395309448242ms +- 0.12215970627618267ms
Iter 13/20: MACs 11.580686863625 -> 1.42193513546875, Params 60.192808 -> 7.2464, latency 32.3137126159668ms +- 0.07397712184963902ms
Iter 14/20: MACs 11.580686863625 -> 1.0553217846875, Params 60.192808 -> 5.360118, latency 28.693135223388673ms +- 0.1303430301052091ms
Iter 15/20: MACs 11.580686863625 -> 0.75849433990625, Params 60.192808 -> 3.792394, latency 20.69975040435791ms +- 0.07094772043898906ms
Iter 16/20: MACs 11.580686863625 -> 0.46923765709375, Params 60.192808 -> 2.357962, latency 19.126435852050783ms +- 0.1028710239503571ms
Iter 17/20: MACs 11.580686863625 -> 0.26916389834375, Params 60.192808 -> 1.34339, latency 17.233325386047362ms +- 0.06552947109856637ms
Iter 18/20: MACs 11.580686863625 -> 0.12086036353125, Params 60.192808 -> 0.569636, latency 16.229191513061522ms +- 0.15013965228417775ms
Iter 19/20: MACs 11.580686863625 -> 0.03155855278125, Params 60.192808 -> 0.142928, latency 16.606309051513673ms +- 0.07111197705444444ms
Iter 20/20: MACs 11.580686863625 -> 0.03155855278125, Params 60.192808 -> 0.142928, latency 16.638975257873536ms +- 0.06173731516679735ms
```



## References

<a id="1">[1]</a> Nisp: Pruning networks using neuron impor- tance score propagation. 

<a id="2">[2]</a> Filter pruning via geometric median for deep con-volutional neural networks acceleration. 

<a id="3">[3]</a> Neuron-level structured pruning using polarization regularizer.  

<a id="4">[4]</a> Pruning Filters for Efficient ConvNets.

<a id="5">[5]</a> Amc: Automl for model compression and ac- 933 celeration on mobile devices.

<a id="6">[6]</a> Hrank: Filter pruning using high-rank feature map.

<a id="7">[7]</a> Soft filter pruning for accelerating deep convolutional 929 neural networks

<a id="8">[8]</a> Resrep: Lossless cnn pruning via decoupling remembering and forgetting.

<a id="9">[9]</a> Gate decorator: Global filter pruning method for accelerating deep convolutional neural networks.

<a id="10">[10]</a> Auto-balanced filter pruning for efficient convolutional neural networks.

<a id="11">[11]</a> Centripetal sgd for pruning very deep convolutional networks with complicated structure

<a id="12">[12]</a> Neural pruning via growing regularization

<a id="13">[13]</a>  Eigendamage: Structured pruning in the kroneckerfactored eigenbasis.
