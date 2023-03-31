# Prunability

## Torchvision 0.13.1
```python
python torchvision_pruning.py
```

#### Outputs:
```
Successful Pruning: 73 Models
 ['alexnet', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14', 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'googlenet', 'inception_v3', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf', 'regnet_y_128gf', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'ssdlite320_mobilenet_v3_large', 'ssd300_vgg16', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn']
```

```
Unsuccessful Pruning: 12 Models
 ['fcos_resnet50_fpn', 'raft_large', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'swin_t', 'swin_s', 'swin_b', 'keypointrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2', 'retinanet_resnet50_fpn_v2']
```

## YOLO v7

This script provides a basic example of pruning YOLOv7, but it does not include any code for fine-tuning. As such, it is intended as a starting point for those interested in exploring YOLOv7 pruning, and further work will be needed to fintune the pruned model's performance after pruning.

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cp yolov7_detect_pruned.py yolov7/

python yolov7_detect_pruned.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```

