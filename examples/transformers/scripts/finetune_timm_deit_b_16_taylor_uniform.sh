torchrun --nproc_per_node=8 finetune.py \
    --model "output/pruned/deit_base_patch16_224_pruned_taylor_uniform.pth" \
    --epochs 300 \
    --batch-size 256 \
    --opt adamw \
    --lr 0.00015 \
    --wd 0.3 \
    --lr-scheduler cosineannealinglr \
    --lr-warmup-method linear \
    --lr-warmup-epochs 0 \
    --lr-warmup-decay 0.033 \
    --amp \
    --label-smoothing 0.11 \
    --mixup-alpha 0.2 \
    --auto-augment ra \
    --clip-grad-norm 1 \
    --ra-sampler \
    --random-erase 0.25 \
    --cutmix-alpha 1.0 \
    --data-path "data/imagenet" \
    --output-dir output/deit_b_16_pruning_taylor_uniform \
    --use_imagenet_mean_std \