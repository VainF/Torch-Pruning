python main.py --mode pretrain --dataset modelnet40 --model dgcnn --lr 0.1 --total-epochs 250 --lr-decay-milestones 100,150,200 --batch-size  32 --output-dir run/modelnet40

python main.py --mode prune --model dgcnn --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --dataset modelnet40  --method group_norm --speed-up 4.0 --soft-rank 0.5 --global --lr 0.01 --total-epochs 100 --lr-decay-milestones 50,80 --batch-size 32 --output-dir run/modelnet40
