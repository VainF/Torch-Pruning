python main.py --mode prune --model resnet56 --batch_size 128 --restore run/pretrain/cifar100_resnet56.pth --pruning_steps 1 --sparsity 0.2 --dataset cifar100  --method dropout  # 40
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/pretrain/cifar100_resnet56.pth --pruning_steps 1 --sparsity 0.4 --dataset cifar100  --method dropout  # 60
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/pretrain/cifar100_resnet56.pth --pruning_steps 1 --sparsity 0.6 --dataset cifar100  --method dropout   # 80
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/pretrain/cifar100_resnet56.pth --pruning_steps 1 --sparsity 0.8 --dataset cifar100  --method dropout   # 90

python main.py --mode prune --model vgg19 --batch_size 128 --restore run/pretrain/cifar100_vgg19.pth --pruning_steps 1 --sparsity 0.2 --dataset cifar100  --method dropout  
python main.py --mode prune --model vgg19 --batch_size 128 --restore run/pretrain/cifar100_vgg19.pth --pruning_steps 1 --sparsity 0.4 --dataset cifar100  --method dropout  
python main.py --mode prune --model vgg19 --batch_size 128 --restore run/pretrain/cifar100_vgg19.pth --pruning_steps 1 --sparsity 0.6 --dataset cifar100  --method dropout  
python main.py --mode prune --model vgg19 --batch_size 128 --restore run/pretrain/cifar100_vgg19.pth --pruning_steps 1 --sparsity 0.8 --dataset cifar100  --method dropout  #

python main.py --mode prune --model densenet121 --batch_size 128 --restore run/pretrain/cifar100_densenet121.pth --pruning_steps 1 --sparsity 0.2 --dataset cifar100  --method dropout  
python main.py --mode prune --model densenet121 --batch_size 128 --restore run/pretrain/cifar100_densenet121.pth --pruning_steps 1 --sparsity 0.4 --dataset cifar100  --method dropout  
python main.py --mode prune --model densenet121 --batch_size 128 --restore run/pretrain/cifar100_densenet121.pth --pruning_steps 1 --sparsity 0.6 --dataset cifar100  --method dropout  
python main.py --mode prune --model densenet121 --batch_size 128 --restore run/pretrain/cifar100_densenet121.pth --pruning_steps 1 --sparsity 0.8 --dataset cifar100  --method dropout  #
#

python main.py --mode prune --model inceptionv4 --batch_size 128 --restore run/pretrain/cifar100_inceptionv4.pth --pruning_steps 1 --sparsity 0.2 --dataset cifar100  --method dropout  
python main.py --mode prune --model inceptionv4 --batch_size 128 --restore run/pretrain/cifar100_inceptionv4.pth --pruning_steps 1 --sparsity 0.4 --dataset cifar100  --method dropout  
python main.py --mode prune --model inceptionv4 --batch_size 128 --restore run/pretrain/cifar100_inceptionv4.pth --pruning_steps 1 --sparsity 0.6 --dataset cifar100  --method dropout  
python main.py --mode prune --model inceptionv4 --batch_size 128 --restore run/pretrain/cifar100_inceptionv4.pth --pruning_steps 1 --sparsity 0.8 --dataset cifar100  --method dropout  #
#

python main.py --mode prune --model googlenet --batch_size 128 --restore run/pretrain/cifar100_googlenet.pth --pruning_steps 1 --sparsity 0.2 --dataset cifar100  --method dropout  
python main.py --mode prune --model googlenet --batch_size 128 --restore run/pretrain/cifar100_googlenet.pth --pruning_steps 1 --sparsity 0.4 --dataset cifar100  --method dropout  
python main.py --mode prune --model googlenet --batch_size 128 --restore run/pretrain/cifar100_googlenet.pth --pruning_steps 1 --sparsity 0.6 --dataset cifar100  --method dropout  
python main.py --mode prune --model googlenet --batch_size 128 --restore run/pretrain/cifar100_googlenet.pth --pruning_steps 1 --sparsity 0.8 --dataset cifar100  --method dropout  
