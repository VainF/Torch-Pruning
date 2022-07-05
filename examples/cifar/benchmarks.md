# Benchmarks (Under Development)


## CIFAR-100 Scratch Training

|dataset|network|params|top1 Acc|epoch(lr = 0.1)|epoch(lr = 0.01)|epoch(lr = 0.004)|epoch(lr = 0.0008)|total epoch|
|:-----:|:-----:|:----:|:------:|:-------------:|:--------------:|:---------------:|:----------------:|:---------:|
|cifar100|vgg19_bn|20.09M|74.17|120|30|30|20|200|
|cifar100|resnet110|1.74M|74.13|120|30|30|20|200|
|cifar100|densenet201|18.28M||120|30|30|20|200|
|cifar100|googlenet|6.2M||120|30|30|20|200|
|cifar100|inceptionv4|41.3M||120|30|30|20|200|
|cifar100|vit|9.57|

## CIFAR-100 Pruning


|Network | Method | Strategy| Pruned Acc| Original Acc | FLOPs | Speedup |
|:-----:|:-----:|:----:|:------:|:------:|:-------------:|:--------------:| 
|ResNet110|ThiNet| | 
|ResNet110|SSS| |
|ResNet110|IE| | 
|ResNet110|HetConv|
|ResNet110|Meta| | 
|ResNet110|GBN| | 
|ResNet110|GroupFisher| | 
|ResNet110|Ours| |
|
|VGG19|ISTA | 
|VGG19|SFP |
|VGG19|SSS | 
|VGG19|AOFP | 
|VGG19|FPGM | 
|VGG19|IE | |  
|VGG19|GroupFisher | 
|VGG19|Ours |  | 
|
|DenseNet201|AMC |
|DenseNet201|Meta |
|DenseNet201|GroupFisher | 
|DenseNet201|Ours | 
|
|ResNext50|SSS | 
|ResNext50|GroupFisher | 
|ResNext50|Ours | 
| 
| ViT | SSS
| ViT | GroupFisher
| ViT | Ours
|
| Swin-T | SSS
| Swin-T | GroupFisher 
| Swin-T | Ours
| 
| PoolFormer | SSS
| PoolFormer | GroupFisher
| PoolFormer | Ours


## ImageNet Pruning

|Network | Method | Strategy| Pruned Acc| Original Acc | FLOPs | Speedup |
|:-----:|:-----:|:----:|:------:|:------:|:-------------:|:--------------:| 
|ResNet50|ThiNet| | 74.03 | 75.30 | 2.58 | 1.13 |
|ResNet50|SSS| | 75.44 | 76.12 | 3.47 | - |
|ResNet50|IE| | 76.43 | 76.18 | 3.27 | - |
|ResNet50|HetConv| | 76.16 | 76.16 | 2.85 | - |
|ResNet50|Meta| | 76.20 | 76.60 | 3.0 | - |
|ResNet50|GBN| | 76.19 | 75.85 | 2.43 | - |
|ResNet50|GroupFisher| | 76.95 | 76.79 | 3.06 | 1.30 |
|ResNet50|Ours| |
|
|ResNet101|ISTA | | 75.27 | 76.40 | 4.47 | - | 
|ResNet101|SFP | |  77.51 |  77.37 |  4.51 |  - | 
|ResNet101|SSS | |  75.44  | 76.40 | 3.47 |  -
|ResNet101|AOFP | |  76.40 |  76.63 |  3.89 |  - | 
|ResNet101|FPGM | |  77.32 |  77.37 |  4.51 |  - | 
|ResNet101|IE | |   77.35 |  77.37 |  4.70 |  - | 
|ResNet101|GroupFisher | |  78.33 |  78.29 |  3.90 |  1.50
|ResNet101|Ours |  | 
|
|MobileNetv2|AMC | | 70.80 | 71.80 | 0.22 | -
|MobileNetv2|Meta | | 72.70 | 74.70 | 0.29 | -
|MobileNetv2|GroupFisher | | 73.42 | 75.74 | 0.29 | 1.84
|MobileNetv2|Ours | 
|
|ResNext50|SSS | | 74.98 | 77.57 | 2.43 | - |
|ResNext50|GroupFisher | | 77.53 | 77.97 | 2.11 | 1.57 |
|ResNext50|Ours | 
| 
| ViT | SSS
| ViT | GroupFisher
| ViT | Ours
|
| Swin-T | SSS
| Swin-T | GroupFisher 
| Swin-T | Ours
| 
| PoolFormer | SSS
| PoolFormer | GroupFisher
| PoolFormer | Ours

## RNN Pruning

## GCN Pruning

