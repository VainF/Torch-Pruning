torchrun --nproc_per_node=8 train.py\
--dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
--lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1