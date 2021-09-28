# Torch-Pruning: Pruning channels for model acceleration
<div align="center">
<img src="assets/intro.jpg" width="45%">
</div>
Torch-Pruning is a pytorch toolbox for structured neural network pruning. Different from the pruning-by-masking functions in pytorch (unstructured), this toolbox removes entire channels from neural networks for acceleration. Torch-Pruning will automatically detect and handle layer dependencies during pruning. Without too much human effort, it is able to handle various network architectures like DenseNet, ResNet and DeepLab.

### **Features:**
* Channel pruning for CNNs (ResNet, DenseNet, Deeplab, etc.) and Transformers (Bert, contributed by [@horseee](https://github.com/horseee))
* Graph Tracing and automatic dependency maintaining.
* Supported modules: Conv, Linear, BatchNorm, Transposed Conv, PReLU and [customized modules](https://github.com/VainF/Torch-Pruning/blob/master/examples/customized_layer.py).
* Supported operations: split, concatenation, skip connection, flatten, etc.
* Pruning strategies: Random, L1, L2, etc.

## Installation

```bash
pip install torch_pruning # v0.2.7
```

## How it works

Torch-Pruning will forward your model with a fake inputs and collect layer information just like ``torch.jit``. A dependency graph is established to describe the computational graph and layer relations. As pruning a certain layer may affect several different layers (see Quick Start), the dependecy will propogate your pruning operation to other layers automatically and provide a `PruningPlan`. All pruning indices will be correctly transformed if there is ``torch.split`` or ``torch.cat`` in your models.

|  Dependency           |  Visualization  |  Example   |
| :------------------:  | :------------:  | :-----:    |
|    Conv-Conv          |  <img src="assets/conv-conv.png" width="80%"> | AlexNet  |
|    Conv-FC (Global Pooling or Flatten) |  <img src="assets/conv-fc.png" width="80%">   | ResNet, VGG    |  
|    Skip Connection    | <img src="assets/residual.png" width="80%">   | ResNet
|    Concatenation      | <img src="assets/concat.png" width="80%">     | DenseNet, ASPP |
|    Split              | <img src="assets/split.png" width="80%">      | torch.chunk |

**Known Issues**: 

* When groups>1, only depthwise conv is supported, i.e. `groups`=`in_channels`=`out_channels`. 
* Customized operations will be treated as element-wise op, e.g. subclass of `torch.autograd.Function`. 

## Quickstart

### 1. A minimal example 

```python
import torch
from torchvision.models import resnet18
import torch_pruning as tp

model = resnet18(pretrained=True).eval()

# 1. setup strategy (L1 Norm)
strategy = tp.strategy.L1Strategy() # or tp.strategy.RandomStrategy()

# 2. build layer dependency for resnet18
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# 3. get a pruning plan from the dependency graph.
pruning_idxs = strategy(model.conv1.weight, amount=0.4) # or manually selected pruning_idxs=[2, 6, 9, ...]
pruning_plan = DG.get_pruning_plan( model.conv1, tp.prune_conv, idxs=pruning_idxs )
print(pruning_plan)

# 4. execute this plan (prune the model)
pruning_plan.exec()
```

Pruning the resnet.conv1 will affect several layers. Let's inspect the pruning plan (with pruning_idxs=[2, 6, 9]):

```
-------------
[ <DEP: prune_conv => prune_conv on conv1 (Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))>, Index=[2, 6, 9], NumPruned=441]
[ <DEP: prune_conv => prune_batchnorm on bn1 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))>, Index=[2, 6, 9], NumPruned=6]
[ <DEP: prune_batchnorm => _prune_elementwise_op on _ElementWiseOp()>, Index=[2, 6, 9], NumPruned=0]
[ <DEP: _prune_elementwise_op => _prune_elementwise_op on _ElementWiseOp()>, Index=[2, 6, 9], NumPruned=0]
[ <DEP: _prune_elementwise_op => prune_related_conv on layer1.0.conv1 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))>, Index=[2, 6, 9], NumPruned=1728]
[ <DEP: _prune_elementwise_op => _prune_elementwise_op on _ElementWiseOp()>, Index=[2, 6, 9], NumPruned=0]
[ <DEP: _prune_elementwise_op => prune_batchnorm on layer1.0.bn2 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))>, Index=[2, 6, 9], NumPruned=6]
[ <DEP: prune_batchnorm => prune_conv on layer1.0.conv2 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))>, Index=[2, 6, 9], NumPruned=1728]
[ <DEP: _prune_elementwise_op => _prune_elementwise_op on _ElementWiseOp()>, Index=[2, 6, 9], NumPruned=0]
[ <DEP: _prune_elementwise_op => prune_related_conv on layer1.1.conv1 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))>, Index=[2, 6, 9], NumPruned=1728]
[ <DEP: _prune_elementwise_op => _prune_elementwise_op on _ElementWiseOp()>, Index=[2, 6, 9], NumPruned=0]
[ <DEP: _prune_elementwise_op => prune_batchnorm on layer1.1.bn2 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))>, Index=[2, 6, 9], NumPruned=6]
[ <DEP: prune_batchnorm => prune_conv on layer1.1.conv2 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))>, Index=[2, 6, 9], NumPruned=1728]
[ <DEP: _prune_elementwise_op => _prune_elementwise_op on _ElementWiseOp()>, Index=[2, 6, 9], NumPruned=0]
[ <DEP: _prune_elementwise_op => prune_related_conv on layer2.0.conv1 (Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False))>, Index=[2, 6, 9], NumPruned=3456]
[ <DEP: _prune_elementwise_op => prune_related_conv on layer2.0.downsample.0 (Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False))>, Index=[2, 6, 9], NumPruned=384]
11211 parameters will be pruned
-------------
```

Tip: please remember to save the whole model object (weights+architecture) rather than model weights only:

```python
# save a pruned model
# torch.save(model.state_dict(), 'model.pth') # weights only
torch.save(model, 'model.pth') # obj (arch + weights), recommended.

# load a pruned model
model = torch.load('model.pth') # no load_state_dict
```

### 2. Low-level pruning functions

In absence of DependencyGraph, we have to manually handle the broken dependencies layer by layer.

```python
tp.prune_conv( model.conv1, idxs=[2,6,9] )

# fix the broken dependencies manually
tp.prune_batchnorm( model.bn1, idxs=[2,6,9] )
tp.prune_related_conv( model.layer2[0].conv1, idxs=[2,6,9] )
...
```

### 3. Customized Layers

Please refer to [examples/customized_layer.py](https://github.com/VainF/Torch-Pruning/blob/master/examples/customized_layer.py).

### 4. Rounding channels for device-friendly network pruning
You can round the channels by passing a `round_to` parameter to strategy. For example, the following script will round the number of channels to 16xN (e.g., 16, 32, 48, 64).
```python
strategy = tp.strategy.L1Strategy()
pruning_idxs = strategy(model.conv1.weight, amount=0.2, round_to=16)
```
Please refer to [https://github.com/VainF/Torch-Pruning/issues/38](https://github.com/VainF/Torch-Pruning/issues/38) for more details.

### 5. Example: pruning ResNet18 on Cifar10

#### 5.1. Scratch training
```bash
cd examples
python prune_resnet18_cifar10.py --mode train # 11.1M, Acc=0.9248
```

#### 5.2. Pruning and fintuning
```bash
python prune_resnet18_cifar10.py --mode prune --round 1 --total_epochs 30 --step_size 20 # 4.5M, Acc=0.9229
python prune_resnet18_cifar10.py --mode prune --round 2 --total_epochs 30 --step_size 20 # 1.9M, Acc=0.9207
python prune_resnet18_cifar10.py --mode prune --round 3 --total_epochs 30 --step_size 20 # 0.8M, Acc=0.9176
python prune_resnet18_cifar10.py --mode prune --round 4 --total_epochs 30 --step_size 20 # 0.4M, Acc=0.9102
python prune_resnet18_cifar10.py --mode prune --round 5 --total_epochs 30 --step_size 20 # 0.2M, Acc=0.9011
...
```

## Layer Dependency

During structured pruning, we need to maintain the channel consistency between different layers. 

### A Simple Case

<img src="assets/dep1.png" width="80%">

### More Complicated Cases

the layer dependency becomes much more complicated when the model contains skip connections or concatenations. 

#### Residual Block: 
<img src="assets/dep2.png" width="80%">

#### Concatenation: 
<img src="assets/dep3.png" width="80%">

See paper [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) for more details.


