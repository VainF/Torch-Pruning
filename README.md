<div align="center"> <h1>Torch-Pruning <br> <h3>Towards Any Structural Pruning<h3> </h1> </div>
<div align="center">
<img src="assets/intro.jpg" width="45%">
</div>

Torch-Pruning (TP) is a general-purpose library for structural network pruning, which supports a large variaty of nerual networks like Vision Transformers, ResNet, DenseNet, RegNet, ResNext, FCN, DeepLab, VGG, etc. Please refer to [tests/test_torchvision_models.py](tests/test_torchvision_models.py) for more prunable models. Different from the [torch.nn.utils.prune](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) that only zeroizes parameters, Torch-Pruning, powered by a graph algorithm ``DepGraph``, will physically removes parameters and sub-structures from your models. 

Please refer to our preprint paper for more technical details: [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900)

### **Features:**
* Structural (Channel) pruning for [CNNs](tests/test_torchvision_models.py) (e.g. ResNet, DenseNet, Deeplab) and [Transformers](tests/test_torchvision_models.py) (e.g. ViT)
* High-level pruners: MagnitudePruner, BNScalePruner, GroupPruner, RandomPruner, etc.
* Graph Tracing and dependency modeling.
* Supported modules: Conv, Linear, BatchNorm, LayerNorm, Transposed Conv, PReLU, Embedding, MultiheadAttention, nn.Parameters and [customized modules](tests/test_customized_layer.py).
* Supported operations: split, concatenation, skip connection, flatten, all element-wise ops, etc.
* [Low-level pruning functions](https://github.com/VainF/Torch-Pruning/blob/master/torch_pruning/pruner/function.py)
* [Benchmarks](benchmarks) and [tutorials](tutorials)

### **Plans:**
* More high-level pruners like FisherPruner, SoftPruner, GeometricPruner, etc.
* Support more Transformers like Vision Transformers (:heavy_check_mark:), Swin Transformers, PoolFormers.
* More standard layers: GroupNorm, InstanceNorm, Shuffle Layers, etc.
* Examples for GNNs and RNNs.
* Pruning benchmarks for CIFAR, ImageNet and COCO.
* Block/Layer/Depth Pruning

## Installation
```bash
pip install torch-pruning # v1.0.0
```
or
```bash
git clone https://github.com/VainF/Torch-Pruning.git
```

## Quickstart
  
Here we provide a quick start for Torch-Pruning. More explained details can be found in [tutorals](./tutorials/)

### 0. How it works

Dependency emerges in complicated network structures, which forces a group of parameters to be pruned simultaneouly. This works provides an automatical mechanism to group parameters with inter-depenedency, so that they can be correctly removed for acceleration. To be exact, Torch-Pruning will forward your model with a fake input and trace the network to establish a dependency graph, recording the dependency between layers. When you prune a single layer, Torch-pruning will also group those coupled layers by returning a `Group`. All pruning indices will be automatically transformed and aligned if there are operations like ``torch.split`` or ``torch.cat``. We illustrate some dependencies in modern nerual networks as the following, where all highlighted channels and parameters will be removed together.

<div align="center">
<img src="assets/dep.png" width="100%">
</div>

### 1. A minimal example

```python
import torch
from torchvision.models import resnet18
import torch_pruning as tp

model = resnet18(pretrained=True).eval()

# 1. build dependency graph for resnet18
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# 2. Specify the to-be-pruned channels. Here we prune those channels indexed by [2, 6, 9].
pruning_idxs = [2, 6, 9]
pruning_group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs )

# 3. prune all grouped layer that is coupled with model.conv1 (included).
if DG.check_pruning_group(pruning_group): # avoid full pruning, i.e., channels=0.
    pruning_group.exec()

# 4. save & load the pruned model 
torch.save(model, 'model.pth') # save the model object
model_loaded = torch.load('model.pth') # no load_state_dict
```
This example shows the basic pipeline of pruning with DependencyGraph. Note that resnet.conv1 is coupled with several layers. Let's inspect the group and check how a pruning operation "triggers" other ones. 

```
--------------------------------
          Pruning Group
--------------------------------
[0] [DEP] prune_out_channels on conv1 (Conv2d(3, 61, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)) => prune_out_channels on conv1 (Conv2d(3, 61, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)), #Pruned=3
[1] [DEP] prune_out_channels on conv1 (Conv2d(3, 61, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)) => prune_out_channels on bn1 (BatchNorm2d(61, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), #Pruned=3
[2] [DEP] prune_out_channels on bn1 (BatchNorm2d(61, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) => prune_out_channels on _ElementWiseOp(ReluBackward0), #Pruned=3
[3] [DEP] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_out_channels on _ElementWiseOp(MaxPool2DWithIndicesBackward0), #Pruned=3
[4] [DEP] prune_out_channels on _ElementWiseOp(MaxPool2DWithIndicesBackward0) => prune_out_channels on _ElementWiseOp(AddBackward0), #Pruned=3
[5] [DEP] prune_out_channels on _ElementWiseOp(MaxPool2DWithIndicesBackward0) => prune_in_channels on layer1.0.conv1 (Conv2d(61, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), #Pruned=3
[6] [DEP] prune_out_channels on _ElementWiseOp(AddBackward0) => prune_out_channels on layer1.0.bn2 (BatchNorm2d(61, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), #Pruned=3
[7] [DEP] prune_out_channels on _ElementWiseOp(AddBackward0) => prune_out_channels on _ElementWiseOp(ReluBackward0), #Pruned=3
[8] [DEP] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_out_channels on _ElementWiseOp(AddBackward0), #Pruned=3
[9] [DEP] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_in_channels on layer1.1.conv1 (Conv2d(61, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), #Pruned=3
[10] [DEP] prune_out_channels on _ElementWiseOp(AddBackward0) => prune_out_channels on layer1.1.bn2 (BatchNorm2d(61, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), #Pruned=3
[11] [DEP] prune_out_channels on _ElementWiseOp(AddBackward0) => prune_out_channels on _ElementWiseOp(ReluBackward0), #Pruned=3
[12] [DEP] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_in_channels on layer2.0.downsample.0 (Conv2d(61, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)), #Pruned=3
[13] [DEP] prune_out_channels on _ElementWiseOp(ReluBackward0) => prune_in_channels on layer2.0.conv1 (Conv2d(61, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)), #Pruned=3
[14] [DEP] prune_out_channels on layer1.1.bn2 (BatchNorm2d(61, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) => prune_out_channels on layer1.1.conv2 (Conv2d(64, 61, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), #Pruned=3
[15] [DEP] prune_out_channels on layer1.0.bn2 (BatchNorm2d(61, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) => prune_out_channels on layer1.0.conv2 (Conv2d(64, 61, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), #Pruned=3
--------------------------------
```

### 2. High-level Pruners

Based on DependencyGraph, we provide some high-level pruners in this repo for easy pruning. You can specify the channel sparsity to prune the whole model and fintune it using your own training code. Please refer to [tests/test_pruner.py](tests/test_pruner.py) for more details. More examples can be found in [benchmarks/main.py](benchmarks/main.py). 

```python
import torch
from torchvision.models import resnet18
import torch_pruning as tp

model = resnet18(pretrained=True)

# Importance criteria
example_inputs = torch.randn(1, 3, 224, 224)
imp = tp.importance.MagnitudeImportance(p=2)

ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m) # DO NOT prune the final classifier!

iterative_steps = 5 # progressive pruning
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    iterative_steps=iterative_steps,
    ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers,
)

base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
for i in range(iterative_steps):
    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # finetune your model here
    # finetune(model)
    # ...
```

### 3. Low-level pruning functions

Is is also possible to prune your model manually with low-level functions. But it would be quite effort-cunsuming since you need to carefully handle the dependency.

```python
tp.prune_conv_out_channels( model.conv1, idxs=[2,6,9] )

# fix the broken dependencies manually
tp.prune_batchnorm_out_channels( model.bn1, idxs=[2,6,9] )
tp.prune_conv_in_channels( model.layer2[0].conv1, idxs=[2,6,9] )
...
```

The following pruning functions are available:
```python
tp.prune_conv_out_channels,
tp.prune_conv_in_channels,
tp.prune_depthwise_conv_out_channels,
tp.prune_depthwise_conv_in_channels,
tp.prune_batchnorm_out_channels,
tp.prune_batchnorm_in_channels,
tp.prune_linear_out_channels,
tp.prune_linear_in_channels,
tp.prune_prelu_out_channels,
tp.prune_prelu_in_channels,
tp.prune_layernorm_out_channels,
tp.prune_layernorm_in_channels,
tp.prune_embedding_out_channels,
tp.prune_embedding_in_channels,
tp.prune_parameter_out_channels,
tp.prune_parameter_in_channels,
tp.prune_multihead_attention_out_channels,
tp.prune_multihead_attention_in_channels,
```

### 4. Customized Layers

Please refer to [tests/test_customized_layer.py](https://github.com/VainF/Torch-Pruning/blob/master/tests/test_customized_layer.py).

### 5. Benchmarks

Our results on {ResNet-56 / CIFAR-10 / 2.00x}

| Method | Base (%) | Pruned (%) | $\Delta$ Acc (%) | Speed Up |
|:--    |:--:  |:--:    |:--: |:--:      |
| NIPS [[1]](#1)  | -    | -      |-0.03 | 1.76x    |
| Geometric [[2]](#2) | 93.59 | 93.26 | -0.33 | 1.70x |
| Polar [[3]](#3)  | 93.80 | 93.83 | +0.03 |1.88x |
| CP  [[4]](#4)   | 92.80 | 91.80 | -1.00 |2.00x |
| AMC [[5]](#5)   | 92.80 | 91.90 | -0.90 |2.00x |
| HRank [[6]](#6) | 93.26 | 92.17 | -0.09 |2.00x |
| SFP  [[7]](#7)  | 93.59 | 93.36 | +0.23 |2.11x |
| ResRep [[8]](#8) | 93.71 | 93.71 | +0.00 |2.12x |
||
| Ours-L1 | 93.53 | 92.93 | -0.60 | 2.12x |
| Ours-BN | 93.53 | 93.29 | -0.24 | 2.12x |
| Ours-Group | 93.53 | *93.91 | +0.38 | 2.13x |

Please refer to [benchmarks](benchmarks) for more details.

## Citation
```
@article{fang2023depgraph,
  title={DepGraph: Towards Any Structural Pruning},
  author={Fang, Gongfan and Ma, Xinyin and Song, Mingli and Mi, Michael Bi and Wang, Xinchao},
  journal={arXiv preprint arXiv:2301.12900},
  year={2023}
}
```
