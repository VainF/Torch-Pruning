python prune_timm_vit.py \
    --model_name vit_base_patch16_224 \
    --pruning_type hessian \
    --pruning_ratio 0.5 \
    --taylor_batchs 10 \
    --test_accuracy \
    --data_path data/imagenet \
    --train_batch_size 64 \
    --val_batch_size 64 \
    --save_as output/pruned/vit_base_patch16_224_pruned_hessian_uniform.pth \