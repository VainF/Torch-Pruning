python prune_timm_models.py 0.875 --name mobilenetv2_100 --exp_name mbv2_0875

torchrun --nproc_per_node=8 --master_port=25678 finetune_timm_models.py \
        --path /lmy/datasets/imagenet \
        --pt_filepath exp/mbv2_0875/pruned_checkpoint.pt \
        --exp_name mbv2_0875 \
        --batch_size 256 \
        --workers 24 \