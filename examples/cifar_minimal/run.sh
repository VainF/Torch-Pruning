#python prune_resnet18_cifar10.py --mode train # 11.1M, Acc=0.9248

python prune_resnet18_cifar10.py --mode prune --round 1 --total_epochs 1 --step_size 20 # 4.5M, Acc=0.9229
python prune_resnet18_cifar10.py --mode prune --round 2 --total_epochs 1 --step_size 20 # 1.9M, Acc=0.9207
python prune_resnet18_cifar10.py --mode prune --round 3 --total_epochs 1 --step_size 20 # 0.8M, Acc=0.9176
python prune_resnet18_cifar10.py --mode prune --round 4 --total_epochs 1 --step_size 20 # 0.4M, Acc=0.9102
python prune_resnet18_cifar10.py --mode prune --round 5 --total_epochs 1 --step_size 20 # 0.2M, Acc=0.9011