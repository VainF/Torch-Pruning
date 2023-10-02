python prune_timm_vit.py \
    --model_name vit_base_patch16_224 \
    --pruning_type taylor \
    --pruning_ratio 0.6 \
    --taylor_batchs 10 \
    --data_path data/imagenet \
    --train_batch_size 64 \
    --val_batch_size 64 \
    --save_as output/pruned/vit_base_patch16_224_pruned_taylor_uniform.pth \
    --global_pruning