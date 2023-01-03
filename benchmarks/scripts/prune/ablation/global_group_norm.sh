python main.py --mode prune --model resnet56 --batch_size 128 --restore run/cifar100/pretrain/cifar100_resnet56.pth --dataset cifar100  --method group_norm --speed_up 1.5 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/cifar100/pretrain/cifar100_resnet56.pth --dataset cifar100  --method group_norm --speed_up 3.0 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/cifar100/pretrain/cifar100_resnet56.pth --dataset cifar100  --method group_norm --speed_up 6.0 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/cifar100/pretrain/cifar100_resnet56.pth --dataset cifar100  --method group_norm --speed_up 12.0 --soft_rank 0.25 --global_pruning

python main.py --mode prune --model vgg19 --batch_size 128 --restore run/cifar100/pretrain/cifar100_vgg19.pth --dataset cifar100  --method group_norm --speed_up 1.5 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model vgg19 --batch_size 128 --restore run/cifar100/pretrain/cifar100_vgg19.pth --dataset cifar100  --method group_norm --speed_up 3.0 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model vgg19 --batch_size 128 --restore run/cifar100/pretrain/cifar100_vgg19.pth --dataset cifar100  --method group_norm --speed_up 6.0 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model vgg19 --batch_size 128 --restore run/cifar100/pretrain/cifar100_vgg19.pth --dataset cifar100  --method group_norm --speed_up 12.0 --soft_rank 0.25 --global_pruning 

python main.py --mode prune --model densenet121 --batch_size 128 --restore run/cifar100/pretrain/cifar100_densenet121.pth --dataset cifar100  --method group_norm --speed_up 1.5 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model densenet121 --batch_size 128 --restore run/cifar100/pretrain/cifar100_densenet121.pth --dataset cifar100  --method group_norm --speed_up 3.0 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model densenet121 --batch_size 128 --restore run/cifar100/pretrain/cifar100_densenet121.pth --dataset cifar100  --method group_norm --speed_up 6.0 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model densenet121 --batch_size 128 --restore run/cifar100/pretrain/cifar100_densenet121.pth --dataset cifar100  --method group_norm --speed_up 12.0 --soft_rank 0.25 --global_pruning 

python main.py --mode prune --model mobilenetv2 --batch_size 128 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --dataset cifar100  --method group_norm --speed_up 1.5 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model mobilenetv2 --batch_size 128 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --dataset cifar100  --method group_norm --speed_up 3.0 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model mobilenetv2 --batch_size 128 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --dataset cifar100  --method group_norm --speed_up 6.0 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model mobilenetv2 --batch_size 128 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --dataset cifar100  --method group_norm --speed_up 12.0 --soft_rank 0.25 --global_pruning 

python main.py --mode prune --model googlenet --batch_size 128 --restore run/cifar100/pretrain/cifar100_googlenet.pth --dataset cifar100  --method group_norm --speed_up 1.5 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model googlenet --batch_size 128 --restore run/cifar100/pretrain/cifar100_googlenet.pth --dataset cifar100  --method group_norm --speed_up 3.0 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model googlenet --batch_size 128 --restore run/cifar100/pretrain/cifar100_googlenet.pth --dataset cifar100  --method group_norm --speed_up 6.0 --soft_rank 0.25 --global_pruning
python main.py --mode prune --model googlenet --batch_size 128 --restore run/cifar100/pretrain/cifar100_googlenet.pth --dataset cifar100  --method group_norm --speed_up 12.0 --soft_rank 0.25 --global_pruning 
