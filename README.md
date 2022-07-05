<div align="center"> <h1>Torch-Pruning <br> <h3>Pruning channels for model acceleration<h3> </h1> </div>
<div align="center">
<img src="assets/intro.jpg" width="45%">
</div>

Torch-Pruning is a general-purpose library for structured channel pruning, which supports a large variaty of nerual networks like ResNet, DenseNet, RegNet, ResNext, FCN, DeepLab, etc.


Pruning is a popular approach to reduce the heavy computational cost of neural networks by removing redundancies. Existing pruning methods prune networks in a case-by-case way, i.e., writing **different code for different models**. However, with the network designs being more and more complicated, applying traditional pruning algorithms modern neural networks is very difficult. One of the most prominent problems in pruning comes from layer dependencies, where several layers are coupled and must be pruned simultaneously to guarantee the correctness of networks. This project provides the ability of detecting and handling layer dependencies. 

### **Features:**
* Channel pruning for [CNNs](tests/test_torchvision_models.py) (e.g. ResNet, DenseNet, Deeplab) and [Transformers](tests/test_torchvision_models.py) (e.g. ViT)
* Graph Tracing and dependency fixing.
* Supported modules: Conv, Linear, BatchNorm, LayerNorm, Transposed Conv, PReLU, Embedding, nn.Parameters and [customized modules](tests/test_customized_layer.py).
* Supported operations: split, concatenation, skip connection, flatten, etc.
* Pruning strategies: Random, L1, L2, etc.
* Low-level pruning [functions](torch_pruning/prune/structured.py)

### Updates:
**02/07/2022** A new version is available in branch [v1.0](https://github.com/VainF/Torch-Pruning/tree/v1.0), which supports Vision Transformers, Group Convolutions, etc.

**24/03/2022** We are drafting a paper to provide more technical details about this repo, which will be released as soon as possible, together with a new version and some practical examples for yolo and other popular networks.
  
## How it works
  
Torch-Pruning will forward your model with a fake inputs and trace the computational graph just like ``torch.jit``. A dependency graph will be established to record the relation coupling between layers. Torch-pruning will collect all affected layers according by propogating your pruning operations through the whole graph, and then return a `PruningPlan` for pruning. All pruning indices will be automatically transformed if there are operations like ``torch.split`` or ``torch.cat``. 
  
## Installation

```bash
pip install torch_pruning # v0.2.7
```
**Known Issues**: 

* When groups>1, only depthwise conv is supported, i.e. `groups`=`in_channels`=`out_channels`. 
* Customized operations will be treated as element-wise op, e.g. subclass of `torch.autograd.Function`. 


## Quickstart
  
### 0. Dependenies

|  Dependency           |  Visualization  |  Example   |
| :------------------:  | :------------:  | :-----:    |
|    Conv-Conv          |  <img src="assets/conv-conv.png" width="80%"> | AlexNet  |
|    Conv-FC (Global Pooling or Flatten) |  <img src="assets/conv-fc.png" width="80%">   | ResNet, VGG    |  
|    Skip Connection    | <img src="assets/residual.png" width="80%">   | ResNet
|    Concatenation      | <img src="assets/concat.png" width="80%">     | DenseNet, ASPP |
|    Split              | <img src="assets/split.png" width="80%">      | torch.chunk |


### 1. A minimal example

```python
import torch
from torchvision.models import resnet18
import torch_pruning as tp

model = resnet18(pretrained=True).eval()

# 1. setup strategy (L1 Norm)
strategy = tp.strategy.L1Strategy() # or tp.strategy.RandomStrategy()

# 2. build dependency graph for resnet18
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# 3. get a pruning plan from the dependency graph.
pruning_idxs = strategy(model.conv1.weight, amount=0.4) # or pruning_idxs=[2, 6, 9, ...]
pruning_plan = DG.get_pruning_plan( model.conv1, tp.prune_conv_out_channel, idxs=pruning_idxs )
print(pruning_plan)

# 4. execute this plan after checking (prune the model)
#    if the plan prunes some channels to zero, 
#    DG.check_pruning plan will return False.
if DG.check_pruning_plan(pruning_plan):
    pruning_plan.exec()
```

Pruning the resnet.conv1 will affect several layers. Let's inspect the pruning plan (with pruning_idxs=[2, 6, 9]). It return the pruning details and the total amount of pruned parameters. You can also customize the metrics following [test_metrics.py](tests/test_metrics.py).

```
--------------------------------
          Pruning Plan
--------------------------------
User pruning:
[ [DEP] ConvOutChannelPruner on conv1 (Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)) => ConvOutChannelPruner on conv1 (Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)), Index=[0, 2, 6], metric={'#params': 441}]

Coupled pruning:
[ [DEP] ConvOutChannelPruner on conv1 (Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)) => BatchnormPruner on bn1 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), Index=[0, 2, 6], metric={'#params': 6}]
[ [DEP] BatchnormPruner on bn1 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) => ElementWiseOpPruner on _ElementWiseOp(ReluBackward0), Index=[0, 2, 6], metric={}]
[ [DEP] ElementWiseOpPruner on _ElementWiseOp(ReluBackward0) => ElementWiseOpPruner on _ElementWiseOp(MaxPool2DWithIndicesBackward0), Index=[0, 2, 6], metric={}]
[ [DEP] ElementWiseOpPruner on _ElementWiseOp(MaxPool2DWithIndicesBackward0) => ElementWiseOpPruner on _ElementWiseOp(AddBackward0), Index=[0, 2, 6], metric={}]
[ [DEP] ElementWiseOpPruner on _ElementWiseOp(MaxPool2DWithIndicesBackward0) => ConvInChannelPruner on layer1.0.conv1 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), Index=[0, 2, 6], metric={'#params': 1728}]
[ [DEP] ElementWiseOpPruner on _ElementWiseOp(AddBackward0) => BatchnormPruner on layer1.0.bn2 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), Index=[0, 2, 6], metric={'#params': 6}]
[ [DEP] ElementWiseOpPruner on _ElementWiseOp(AddBackward0) => ElementWiseOpPruner on _ElementWiseOp(ReluBackward0), Index=[0, 2, 6], metric={}]
[ [DEP] ElementWiseOpPruner on _ElementWiseOp(ReluBackward0) => ElementWiseOpPruner on _ElementWiseOp(AddBackward0), Index=[0, 2, 6], metric={}]
[ [DEP] ElementWiseOpPruner on _ElementWiseOp(ReluBackward0) => ConvInChannelPruner on layer1.1.conv1 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), Index=[0, 2, 6], metric={'#params': 1728}]
[ [DEP] ElementWiseOpPruner on _ElementWiseOp(AddBackward0) => BatchnormPruner on layer1.1.bn2 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), Index=[0, 2, 6], metric={'#params': 6}]
[ [DEP] ElementWiseOpPruner on _ElementWiseOp(AddBackward0) => ElementWiseOpPruner on _ElementWiseOp(ReluBackward0), Index=[0, 2, 6], metric={}]
[ [DEP] ElementWiseOpPruner on _ElementWiseOp(ReluBackward0) => ConvInChannelPruner on layer2.0.downsample.0 (Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)), Index=[0, 2, 6], metric={'#params': 384}]
[ [DEP] ElementWiseOpPruner on _ElementWiseOp(ReluBackward0) => ConvInChannelPruner on layer2.0.conv1 (Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)), Index=[0, 2, 6], metric={'#params': 3456}]
[ [DEP] BatchnormPruner on layer1.1.bn2 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) => ConvOutChannelPruner on layer1.1.conv2 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), Index=[0, 2, 6], metric={'#params': 1728}]
[ [DEP] BatchnormPruner on layer1.0.bn2 (BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) => ConvOutChannelPruner on layer1.0.conv2 (Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)), Index=[0, 2, 6], metric={'#params': 1728}]

Metric Sum: {'#params': 11211}
--------------------------------
```

Tip: please remember to save the whole model object (weights+architecture) after pruning, instead of saving the weights dict:

```python
# save a pruned model
# torch.save(model.state_dict(), 'model.pth') # weights only
torch.save(model, 'model.pth') # obj (arch + weights), recommended.

# load a pruned model
model = torch.load('model.pth') # no load_state_dict
```

### 2. Low-level pruning functions

It is equivalent to make a layer-by-layer fixing using the low-level pruning functions. 

```python
tp.prune_conv_out_channel( model.conv1, idxs=[2,6,9] )

# fix the broken dependencies manually
tp.prune_batchnorm( model.bn1, idxs=[2,6,9] )
tp.prune_conv_in_channel( model.layer2[0].conv1, idxs=[2,6,9] )
...
```

### 3. Group Convs
We provide a tool `tp.helpers.gconv2convs()`  to transform Group Conv to a group of vanilla convs. Please refer to [test_convnext.py](tests/test_convnext.py) for more details.

### 4. Customized Layers

Please refer to [examples/customized_layer.py](https://github.com/VainF/Torch-Pruning/blob/master/examples/customized_layer.py).

### 5. Rounding channels for device-friendly network pruning
You can round the channels by passing a `round_to` parameter to strategy. For example, the following script will round the number of channels to 16xN (e.g., 16, 32, 48, 64).
```python
strategy = tp.strategy.L1Strategy()
pruning_idxs = strategy(model.conv1.weight, amount=0.2, round_to=16)
```
Please refer to [https://github.com/VainF/Torch-Pruning/issues/38](https://github.com/VainF/Torch-Pruning/issues/38) for more details.

### 5. Example: pruning ResNet18 on Cifar10

#### 5.1. Scratch training
```bash
cd examples/cifar_minimal
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


