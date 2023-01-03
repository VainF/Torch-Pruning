# CIFAR-10
python main.py --mode pretrain --dataset cifar10 --model resnet56 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180 
python main.py --mode pretrain --dataset cifar10 --model vgg19 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180 

# CIFAR-100
python main.py --mode pretrain --dataset cifar100 --model resnet56 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180 
python main.py --mode pretrain --dataset cifar100 --model vgg19 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180 
python main.py --mode pretrain --dataset cifar100 --model densenet121 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180 
python main.py --mode pretrain --dataset cifar100 --model googlenet --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180 
python main.py --mode pretrain --dataset cifar100 --model mobilenetv2 --lr 0.05 --total-epochs 200 --lr-decay-milestones 120,150,180
python main.py --mode pretrain --dataset cifar100 --model resnext50 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180 