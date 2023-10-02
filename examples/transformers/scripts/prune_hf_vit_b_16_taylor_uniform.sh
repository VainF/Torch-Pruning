python prune_hf_vit.py \
    --model_name google/vit-base-patch16-224 \
    --pruning_type taylor \
    --pruning_ratio 0.5 \
    --taylor_batchs 10 \
    --data_path data/imagenet \
    --test_accuracy \
    --train_batch_size 64 \
    --val_batch_size 64 \
    --save_as output/pruned/hf_vit_base_patch16_224_pruned_taylor_uniform.pth \