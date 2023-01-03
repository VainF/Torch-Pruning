# Benchmarks (Beta version)


## CIFAR10

#### Pretraining
```python
python main.py --mode pretrain --dataset cifar10 --model resnet56 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180 
```

#### Local pruning with L1 Norm
```python
python main.py --mode prune --model resnet56 --batch-size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method l1 --speed-up 2.0 
```

#### Global pruning with L1 Norm
```python
python main.py --mode prune --model resnet56 --batch-size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method l1 --speed-up 2.0 --global-pruning
```

## ImageNet

* ResNet50 Pruning, L1 Pruner, 2.00 GFlops

```python
python -m torch.distributed.launch --nproc_per_node=4 --master_port 18119 --use_env main_imagenet.py --model resnet50 --epochs 90 --batch-size 64 --lr-step-size 30 --lr 0.01 --prune --method l1 --pretrained --output-dir run/imagenet/resnet50_sl --target-flops 2.00 --cache-dataset --print-freq 100 --workers 16 --data-path PATH_TO_IMAGENET --output-dir PATH_TO_OUTPUT_DIR # &> output.log
```