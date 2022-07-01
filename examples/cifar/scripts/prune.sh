python prune_cifar.py --mode prune --model vit_cifar --batch_size 128 --round 1 --restore_from checkpoints/scratch/cifar100_vit_cifar.pth
python prune_cifar.py --mode prune --model vit_cifar --batch_size 128 --round 2 --restore_from checkpoints/scratch/cifar100_vit_cifar-round1.pth
python prune_cifar.py --mode prune --model vit_cifar --batch_size 128 --round 3 --restore_from checkpoints/scratch/cifar100_vit_cifar-round2.pth
python prune_cifar.py --mode prune --model vit_cifar --batch_size 128 --round 4 --restore_from checkpoints/scratch/cifar100_vit_cifar-round3.pth


python prune_cifar.py --mode prune --model resnet110 --batch_size 128 --round 1 --restore_from checkpoints/scratch/cifar100_resnet110.pth
python prune_cifar.py --mode prune --model resnet110 --batch_size 128 --round 2 --restore_from checkpoints/scratch/cifar100_resnet110-round1.pth
python prune_cifar.py --mode prune --model resnet110 --batch_size 128 --round 3 --restore_from checkpoints/scratch/cifar100_resnet110-round2.pth
python prune_cifar.py --mode prune --model resnet110 --batch_size 128 --round 4 --restore_from checkpoints/scratch/cifar100_resnet110-round3.pth