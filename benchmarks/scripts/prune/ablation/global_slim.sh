python main.py --mode prune --model resnet56 --batch_size 128 --restore run/cifar100/pretrain/cifar100_resnet56.pth --dataset cifar100  --method slim --speed_up 1.5 --reg 1e-4 --global
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/cifar100/pretrain/cifar100_resnet56.pth --dataset cifar100  --method slim --speed_up 3.0 --reg 1e-4 --global
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/cifar100/pretrain/cifar100_resnet56.pth --dataset cifar100  --method slim --speed_up 6.0 --reg 1e-4 --global
python main.py --mode prune --model resnet56 --batch_size 128 --restore run/cifar100/pretrain/cifar100_resnet56.pth --dataset cifar100  --method slim --speed_up 12.0 --reg 1e-4 --global

python main.py --mode prune --model vgg19 --batch_size 128 --restore run/cifar100/pretrain/cifar100_vgg19.pth --dataset cifar100  --method lamp --speed_up 1.5 --reg 1e-4 --global
python main.py --mode prune --model vgg19 --batch_size 128 --restore run/cifar100/pretrain/cifar100_vgg19.pth --dataset cifar100  --method lamp --speed_up 3.0 --reg 1e-4 --global
python main.py --mode prune --model vgg19 --batch_size 128 --restore run/cifar100/pretrain/cifar100_vgg19.pth --dataset cifar100  --method lamp --speed_up 6.0 --reg 1e-4 --global
python main.py --mode prune --model vgg19 --batch_size 128 --restore run/cifar100/pretrain/cifar100_vgg19.pth --dataset cifar100  --method lamp --speed_up 12.0 --reg 1e-4 --global 

python main.py --mode prune --model densenet121 --batch_size 128 --restore run/cifar100/pretrain/cifar100_densenet121.pth --dataset cifar100  --method lamp --speed_up 1.5 --reg 1e-4 --global
python main.py --mode prune --model densenet121 --batch_size 128 --restore run/cifar100/pretrain/cifar100_densenet121.pth --dataset cifar100  --method lamp --speed_up 3.0 --reg 1e-4 --global
python main.py --mode prune --model densenet121 --batch_size 128 --restore run/cifar100/pretrain/cifar100_densenet121.pth --dataset cifar100  --method lamp --speed_up 6.0 --reg 1e-4 --global
python main.py --mode prune --model densenet121 --batch_size 128 --restore run/cifar100/pretrain/cifar100_densenet121.pth --dataset cifar100  --method lamp --speed_up 12.0 --reg 1e-4 --global 

python main.py --mode prune --model mobilenetv2 --batch_size 128 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --dataset cifar100  --method lamp --speed_up 1.5 --reg 1e-4 --global
python main.py --mode prune --model mobilenetv2 --batch_size 128 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --dataset cifar100  --method lamp --speed_up 3.0 --reg 1e-4 --global
python main.py --mode prune --model mobilenetv2 --batch_size 128 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --dataset cifar100  --method lamp --speed_up 6.0 --reg 1e-4 --global
python main.py --mode prune --model mobilenetv2 --batch_size 128 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --dataset cifar100  --method lamp --speed_up 12.0 --reg 1e-4 --global 

python main.py --mode prune --model googlenet --batch_size 128 --restore run/cifar100/pretrain/cifar100_googlenet.pth --dataset cifar100  --method lamp --speed_up 1.5 --reg 1e-4 --global
python main.py --mode prune --model googlenet --batch_size 128 --restore run/cifar100/pretrain/cifar100_googlenet.pth --dataset cifar100  --method lamp --speed_up 3.0 --reg 1e-4 --global
python main.py --mode prune --model googlenet --batch_size 128 --restore run/cifar100/pretrain/cifar100_googlenet.pth --dataset cifar100  --method lamp --speed_up 6.0 --reg 1e-4 --global
python main.py --mode prune --model googlenet --batch_size 128 --restore run/cifar100/pretrain/cifar100_googlenet.pth --dataset cifar100  --method lamp --speed_up 12.0 --reg 1e-4 --global 
