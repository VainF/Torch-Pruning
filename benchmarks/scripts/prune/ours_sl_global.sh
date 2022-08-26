python main.py --mode prune --model resnet56 --batch_size 128 --restore run/pretrain/cifar100_resnet56.pth --dataset cifar100  --method group_sl --speed_up 1.5 --global --reg 2e-5 
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/pretrain/cifar100_resnet56.pth --dataset cifar100  --method group_sl --speed_up 3.0 --global --reg 2e-5 
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/pretrain/cifar100_resnet56.pth --dataset cifar100  --method group_sl --speed_up 6.0 --global --reg 2e-5 
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/pretrain/cifar100_resnet56.pth --dataset cifar100  --method group_sl --speed_up 12.0 --global --reg 2e-5 

python main.py --mode prune --model vgg19 --batch_size 128 --restore run/pretrain/cifar100_vgg19.pth --dataset cifar100  --method group_sl --speed_up 1.5 --global --reg 1e-4
python main.py --mode prune --model vgg19 --batch_size 128 --restore run/pretrain/cifar100_vgg19.pth --dataset cifar100  --method group_sl --speed_up 3.0 --global --reg 1e-4
python main.py --mode prune --model vgg19 --batch_size 128 --restore run/pretrain/cifar100_vgg19.pth --dataset cifar100  --method group_sl --speed_up 6.0 --global --reg 1e-4
python main.py --mode prune --model vgg19 --batch_size 128 --restore run/pretrain/cifar100_vgg19.pth --dataset cifar100  --method group_sl --speed_up 12.0 --global  --reg 1e-4

python main.py --mode prune --model densenet121 --batch_size 128 --restore run/pretrain/cifar100_densenet121.pth --dataset cifar100  --method group_sl --speed_up 1.5 --global --reg 1e-4
python main.py --mode prune --model densenet121 --batch_size 128 --restore run/pretrain/cifar100_densenet121.pth --dataset cifar100  --method group_sl --speed_up 3.0 --global --reg 1e-4
python main.py --mode prune --model densenet121 --batch_size 128 --restore run/pretrain/cifar100_densenet121.pth --dataset cifar100  --method group_sl --speed_up 6.0 --global --reg 1e-4
python main.py --mode prune --model densenet121 --batch_size 128 --restore run/pretrain/cifar100_densenet121.pth --dataset cifar100  --method group_sl --speed_up 12.0 --global  --reg 1e-4

python main.py --mode prune --model inceptionv4 --batch_size 128 --restore run/pretrain/cifar100_inceptionv4.pth --dataset cifar100  --method group_sl --speed_up 1.5 --global --reg 1e-4
python main.py --mode prune --model inceptionv4 --batch_size 128 --restore run/pretrain/cifar100_inceptionv4.pth --dataset cifar100  --method group_sl --speed_up 3.0 --global --reg 1e-4
python main.py --mode prune --model inceptionv4 --batch_size 128 --restore run/pretrain/cifar100_inceptionv4.pth --dataset cifar100  --method group_sl --speed_up 6.0 --global --reg 1e-4
python main.py --mode prune --model inceptionv4 --batch_size 128 --restore run/pretrain/cifar100_inceptionv4.pth --dataset cifar100  --method group_sl --speed_up 12.0 --global  --reg 1e-4

python main.py --mode prune --model googlenet --batch_size 128 --restore run/pretrain/cifar100_googlenet.pth --dataset cifar100  --method group_sl --speed_up 1.5 --global --reg 1e-4
python main.py --mode prune --model googlenet --batch_size 128 --restore run/pretrain/cifar100_googlenet.pth --dataset cifar100  --method group_sl --speed_up 3.0 --global --reg 1e-4
python main.py --mode prune --model googlenet --batch_size 128 --restore run/pretrain/cifar100_googlenet.pth --dataset cifar100  --method group_sl --speed_up 6.0 --global --reg 1e-4
python main.py --mode prune --model googlenet --batch_size 128 --restore run/pretrain/cifar100_googlenet.pth --dataset cifar100  --method group_sl --speed_up 12.0 --global  --reg 1e-4