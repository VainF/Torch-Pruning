python prune_timm_vit.py \
    --model_name deit_base_distilled_patch16_224 \
    --pruning_type taylor \
    --pruning_ratio 0.54 \
    --taylor_batchs 50 \
    --data_path data/imagenet \
    --train_batch_size 64 \
    --val_batch_size 64 \
    --save_as output/pruned/deit_base_patch16_224_pruned_taylor_uniform.pth \
    --use_imagenet_mean_std \