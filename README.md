<div align="center"> <h1>Torch-Pruning <br> <h3>Structural Pruning for Model Acceleration<h3> </h1> </div>
<div align="center">
<img src="assets/intro.jpg" width="45%">
</div>

Torch-Pruning is a general-purpose library for structural network pruning, which supports a large variaty of nerual networks like Vision Transformers, ResNet, DenseNet, RegNet, ResNext, FCN, DeepLab, VGG, etc. Please refer to [tests/test_torchvision_models.py](tests/test_torchvision_models.py) for more details about prunable models.

### **Features:**
* Channel pruning for [CNNs](tests/test_torchvision_models.py) (e.g. ResNet, DenseNet, Deeplab) and [Transformers](tests/test_torchvision_models.py) (e.g. ViT)
* High-level pruners: LocalMagnitudePruner, GlobalMagnitudePruner, BNScalePruner, etc.
* Graph Tracing and dependency fixing.
* Supported modules: Conv, Linear, BatchNorm, LayerNorm, Transposed Conv, PReLU, Embedding, MultiheadAttention, nn.Parameters and [customized modules](tests/test_customized_layer.py).
* Supported operations: split, concatenation, skip connection, flatten, etc.
* Pruning strategies: Random, L1, L2, etc.
* Low-level pruning [functions](torch_pruning/prune/structured.py)

### Updates
**02/07/2022** The latest version is under development in branch [v1.0](https://github.com/VainF/Torch-Pruning/tree/v1.0).

**24/03/2022** We are drafting a paper to provide more technical details about this repo, which will be released as soon as possible, together with a new version and some practical examples for yolo and other popular networks.

### Plans:
* High-level pruners like MagnitudeBasedPruner (:heavy_check_mark:), SensitivityBasedPruner, HessianBasedPruner and [Slimming Pruner (ICCV'17)](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html).
* Support more Transformers like Vision Transformers (:heavy_check_mark:), Swin Transformers, PoolFormers.
* A pruning benchmark on CIFAR100 and ImageNet.
* Some examples in detection and segmentation.
* A paper about this repo: title (now we are here! :turtle:), abstract, introduction, methodology, experiments and conclusion.

## How it works
  
Torch-Pruning will forward your model with a fake inputs and trace the computational graph just like ``torch.jit``. A dependency graph will be established to record the relation coupling between layers. Torch-pruning will collect all affected layers according by propogating your pruning operations through the whole graph, and then return a `PruningClique` for pruning. All pruning indices will be automatically transformed if there are operations like ``torch.split`` or ``torch.cat``. 
  
## Installation

```bash
git clone https://github.com/VainF/Torch-Pruning.git
```

## Quickstart
  
### 0. Dependency

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

# 1. build dependency graph for resnet18
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# 2. Select channels for pruning, here we prune the channels indexed by [2, 6, 9].
pruning_idxs = pruning_idxs=[2, 6, 9, ...]
pruning_group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channel, idxs=pruning_idxs )

# 3. prune all grouped layer that is coupled with model.conv1
if DG.check_pruning_group(pruning_group):
    pruning_group.exec()

# 4. save & load the pruned model 
torch.save(model, 'model.pth') # save the model object
model_loaded = torch.load('model.pth') # no load_state_dict
```

In this example, pruning resnet.conv1 will affect several layers. Let's inspect the pruning group (with pruning_idxs=[2, 6, 9]):

```
--------------------------------
          Pruning Clique
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

### 2. High-level Pruners

We provide some model-level pruners in this repo for convenience. You can specify the channel sparsity to prune the whole model and fintune it using your own training code. Please refer to [tests/test_pruner.py](tests/test_pruner.py) for more details. More examples can be found in [benchmarks/main.py](benchmarks/main.py).

```python
import torch
from torchvision.models import densenet121 as entry
import torch_pruning as tp

model = entry(pretrained=True)
print(model)

ori_size = tp.utils.count_params(model)
example_inputs = torch.randn(1, 3, 224, 224)
imp = tp.importance.MagnitudeImportance(p=2) # L2 norm pruning
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m)

iterative_steps = 5 
pruner = tp.pruner.LocalMagnitudePruner( 
    model,
    example_inputs,
    importance=imp,
    iterative_steps=iterative_steps, # number of iterations
    ch_sparsity=0.5, # channel sparsity
    ignored_layers=ignored_layers, # ignored_layers will not be pruned
)

for i in range(iterative_steps): # iterative pruning
    pruner.step()
    print(
        "  Params: %.2f M => %.2f M"
        % (ori_size / 1e6, tp.utils.count_params(model) / 1e6)
    )
    # Your training code here
    # train(...)
```

### 3. Low-level pruning functions

You can also try to prune your model manually with low-level functions. 

```python
tp.prune_conv_out_channel( model.conv1, idxs=[2,6,9] )

# fix the broken dependencies manually
tp.prune_batchnorm( model.bn1, idxs=[2,6,9] )
tp.prune_conv_in_channel( model.layer2[0].conv1, idxs=[2,6,9] )
...
```

The following pruning functions are available:
```python
tp.prune_conv_in_channel
tp.prune_conv_out_channel
tp.prune_depthwise_conv_out_channels
tp.prune_batchnorm 
tp.prune_linear_in_channel 
tp.prune_linear_out_channel 
tp.prune_prelu
tp.prune_layernorm 
tp.prune_embedding 
tp.prune_parameter
tp.prune_multihead_attention
```

### 4. Customized Layers

Please refer to [tests/test_customized_layer.py](https://github.com/VainF/Torch-Pruning/blob/master/tests/test_customized_layer.py).

# Citation
If you find this repo helpful, please cite:
```
@software{Fang_Torch-Pruning_2022,
  author = {Fang, Gongfan},
  month = {7},
  title = {{Torch-Pruning}},
  url = {https://github.com/VainF/Torch-Pruning},
  version = {0.2.8},
  year = {2022}
}
```
