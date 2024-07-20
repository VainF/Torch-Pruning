python main_gat.py --prune --restore run/ppi/pretrain/best_gat.pth --method group_norm --speed-up 8.0 --soft-rank 0.0 --global

python main_gat.py --prune --restore run/ppi/pretrain/best_gat.pth --method group_sl --speed-up 8.0 --soft-rank 0.5 --global-pruning --reg 1e-4