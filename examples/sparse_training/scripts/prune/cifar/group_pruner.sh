# python main.py --mode pretrain --dataset cifar10 --model resnet56 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180 

python main.py --mode prune --model resnet56 --batch-size 128 --restore run/cifar10/pretrain/cifar10_resnet56.pth --dataset cifar10  --method group_sl --speed-up 2.11 --global-pruning --reg 5e-4
