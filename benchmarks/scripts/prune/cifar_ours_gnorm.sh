python main.py --mode prune --model resnet56 --batch_size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method group_norm --speed_up 2.1 --soft_rank 0.0 --global_pruning
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method group_norm --speed_up 2.5 --soft_rank 0.0 --global_pruning
python main.py --mode prune --model vgg19 --batch_size 128 --restore run/cifar100/pretrain/cifar100_vgg19.pth --dataset cifar100  --method group_norm --speed_up 8.84 --soft_rank 0.1 --global_pruning

#python main.py --mode prune --model resnet56 --batch_size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method group_sl --speed_up 2.1 --soft_rank 0.0 --global_pruning
#python main.py --mode prune --model resnet56 --batch_size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method group_sl --speed_up 2.55 --soft_rank 0.0 --global_pruning
#python main.py --mode prune --model vgg19 --batch_size 128 --restore run/cifar100/pretrain/cifar100_vgg19.pth --dataset cifar100  --method group_sl --speed_up 8.84 --soft_rank 0.1 --global_pruning