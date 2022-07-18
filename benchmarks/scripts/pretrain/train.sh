python main.py --mode train --dataset cifar100 --model resnet56 --lr 0.1 --total_epochs 200 --lr_decay_milestones 120,150,180 
python main.py --mode train --dataset cifar100 --model vgg19 --lr 0.1 --total_epochs 200 --lr_decay_milestones 120,150,180 
python main.py --mode train --dataset cifar100 --model inceptionv4 --lr 0.1 --total_epochs 200 --lr_decay_milestones 120,150,180 
python main.py --mode train --dataset cifar100 --model densenet121 --lr 0.1 --total_epochs 200 --lr_decay_milestones 120,150,180 
python main.py --mode train --dataset cifar100 --model googlenet --lr 0.1 --total_epochs 200 --lr_decay_milestones 120,150,180 
