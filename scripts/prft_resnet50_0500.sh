python prune_timm_models.py 0.5 --name resnet50 --exp_name resnet50_0500

torchrun --nproc_per_node=8 --master_port=25678 finetune_timm_models.py \
        --path /lmy/datasets/imagenet \
        --pt_filepath exp/resnet50_0500/pruned_checkpoint.pt \
        --exp_name resnet50_0500 \
        --batch_size 256 \
        --workers 24 \