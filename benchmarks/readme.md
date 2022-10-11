# Benchmarks (Beta version)


## CIFAR10

#### Pretraining
```python
python main.py --mode pretrain --dataset cifar10 --model resnet56 --lr 0.1 --total_epochs 200 --lr_decay_milestones 120,150,180 
```

#### Local pruning with L1 Norm
```python
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method l1 --speed_up 2.0 
```

#### Global pruning with L1 Norm
```python
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method l1 --speed_up 2.0 --global_pruning
```

## ImageNet
```python
python -m torch.distributed.launch --nproc_per_node=2 --master_port 18119 --use_env main_imagenet.py --model resnet50 --epochs 90 --batch-size 128 --lr-step-size 30 --lr 0.01 --prune --method group_sl --global-pruning --sentinel-perc 0.5 --pretrained --output-dir run/imagenet/resnet50_sl --target-flops 2.04  --sl-epochs 30 --sl-lr 0.01 --sl-lr-step-size 10 --cache-dataset --reg 1e-5 --print-freq 1000 --workers 8
```