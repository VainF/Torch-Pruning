# Benchmarks (Beta version)


## ResNet-56 / CIFAR-10 / 2.00x

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
||
| Ours-L1 | 93.53 | 92.93 | -0.60 | 2.12x |
| Ours-BN | 93.53 | 93.29 | -0.24 | 2.12x |
| Ours-Group | 93.53 | 93.91 | +0.38 | 2.13x |

**Note 1:** $\text{speed up} = \frac{\text{Base MACs}}{\text{Pruned MACs}}$

**Note 2:** Baseline methods are not implemented in this repo, because they may require additional modifications to the standard models and training scripts. We are working to support more algorithms.

**Note 3:** Donwload pretrained resnet-56 to reproduce our results: [dropbox](https://www.dropbox.com/s/lcpwz24gcxmo1a7/cifar10_resnet56.pth?dl=0)

**Note 4:** Training logs are available at [run/prune](run/cifar10/prune). 

#### - Pretraining
```python
python main.py --mode pretrain --dataset cifar10 --model resnet56 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180 
```

#### - L1-Norm Pruner
[Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
```bash
# bash scripts/prune/cifar/l1_norm_pruner.sh
python main.py --mode prune --model resnet56 --batch-size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method l1 --speed-up 2.11 --global-pruning
```

#### - BN Pruner
[Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)
```bash
# bash scripts/prune/cifar/bn_pruner.sh
python main.py --mode prune --model resnet56 --batch-size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method slim --speed-up 2.11 --global-pruning --reg 1e-5
```

#### - Group Pruner (this work)
```bash
# bash scripts/prune/cifar/group_lasso_pruner.sh
python main.py --mode prune --model resnet56 --batch-size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method group_lasso --speed-up 2.11 --global-pruning --reg 5e-4
```

## ResNet50 / ImageNet / 2.00 GMACs

#### - L1 Pruner
```python
python -m torch.distributed.launch --nproc_per_node=4 --master_port 18119 --use_env main_imagenet.py --model resnet50 --epochs 90 --batch-size 64 --lr-step-size 30 --lr 0.01 --prune --method l1 --pretrained --output-dir run/imagenet/resnet50_sl --target-flops 2.00 --cache-dataset --print-freq 100 --workers 16 --data-path PATH_TO_IMAGENET --output-dir PATH_TO_OUTPUT_DIR # &> output.log
```

**More results will be released soon!**

## References

<a id="1">[1]</a> Nisp: Pruning networks using neuron impor- tance score propagation. 

<a id="2">[2]</a> Filter pruning via geometric median for deep con-volutional neural networks acceleration. 

<a id="3">[3]</a> Neuron-level structured pruning using polarization regularizer.  

<a id="4">[4]</a> Pruning Filters for Efficient ConvNets.

<a id="5">[5]</a> Amc: Automl for model compression and ac- 933 celeration on mobile devices.

<a id="6">[6]</a> Hrank: Filter pruning using high-rank feature map.

<a id="7">[7]</a> Soft filter pruning for accelerating deep convolutional 929 neural networks

<a id="8">[8]</a> Resrep: Lossless cnn pruning via decoupling remembering and forgetting.
