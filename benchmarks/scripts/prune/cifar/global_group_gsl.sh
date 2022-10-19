CUDA_VISIBLE_DEVICES=4 python main.py --mode prune --model resnet56 --batch-size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method group_sl --speed-up 2.11 --global-pruning --reg 1e-4

CUDA_VISIBLE_DEVICES=4 python main.py --mode prune --model resnet56 --batch-size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method group_sl --speed-up 2.55 --global-pruning --reg 1e-4 # --sl-restore

CUDA_VISIBLE_DEVICES=4 python main.py --mode prune --model vgg19 --batch-size 128 --restore run/cifar100/pretrain/cifar100_vgg19.pth --dataset cifar100  --method group_sl --speed-up 8.84 --global-pruning --reg 1e-4 --max-sparsity 0.7
