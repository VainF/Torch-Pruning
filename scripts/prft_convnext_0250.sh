python prune_timm_models.py 0.25 --model_zoo torchvision --name convnext_tiny --exp_name cntiny_0250

torchrun --nproc_per_node=4 --master_port=25678 finetune_timm_models.py \
        --path /lmy/datasets/imagenet \
        --pt_filepath exp/cntiny_0250/pruned_checkpoint.pt \
        --exp_name cntiny_0250 \
        --batch_size 256 \
        --workers 24 \
