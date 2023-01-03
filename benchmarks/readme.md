# Benchmarks (Beta version)


## ResNet-56 / CIFAR-10 / 2.11x

| Method | Base (%) | Pruned (%) | $\Delta$ Acc (%) | Speed Up |
|:--    |:--:  |:--:    |:--: |:--:      |
| NIPS   | -    | -      |-0.03 | 1.76x    |
| Geometric | 93.59 | 93.26 | -0.33 | 1.70x |
| Polar  | 93.80 | 93.83 | +0.03 |1.88x |
| CP     | 92.80 | 91.80 | -1.00 |2.00x |
| AMC    | 92.80 | 91.90 | -0.90 |2.00x |
| HRank  | 93.26 | 92.17 | -0.09 |2.00x |
| SFP    | 93.59 | 93.36 | +0.23 |2.11x |
| ResRep | 93.71 | 93.71 | +0.00 |2.12x |
| Ours-L1 | 93.53 |
| Ours-BN | 93.53 |
| Ours-Group | 93.53 |

Note: $\text{speed up} = \frac{\text{Base MACs}}{\text{Pruned MACs}}$ 

#### Pretraining
```python
python main.py --mode pretrain --dataset cifar10 --model resnet56 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180 
```

#### L1-Norm Pruner
[Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
```bash
# bash scripts/prune/cifar/l1_norm_pruner.sh
python main.py --mode prune --model resnet56 --batch-size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method l1 --speed-up 2.11 --global-pruning
```

#### BN Pruner
[Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)
```bash
# bash scripts/prune/cifar/bn_pruner.sh
python main.py --mode prune --model resnet56 --batch-size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method slim --speed-up 2.11 --global-pruning --reg 1e-5
```

#### Group Pruner
```bash
# bash scripts/prune/cifar/group_lasso_pruner.sh
python main.py --mode prune --model resnet56 --batch-size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method group_lasso --speed-up 2.11 --global-pruning --reg 5e-4
```

## ResNet50 / ImageNet / 2.00 GFLOPs

#### L1 Pruner
```python
python -m torch.distributed.launch --nproc_per_node=4 --master_port 18119 --use_env main_imagenet.py --model resnet50 --epochs 90 --batch-size 64 --lr-step-size 30 --lr 0.01 --prune --method l1 --pretrained --output-dir run/imagenet/resnet50_sl --target-flops 2.00 --cache-dataset --print-freq 100 --workers 16 --data-path PATH_TO_IMAGENET --output-dir PATH_TO_OUTPUT_DIR # &> output.log
```

**More results will be released soon!**
