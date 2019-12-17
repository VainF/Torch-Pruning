import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch_pruning as pruning

# Convolutional Layer
for conv in [ nn.Conv1d( in_channels=32, out_channels=64, kernel_size=3, stride=2 ),
              nn.Conv2d( in_channels=32, out_channels=64, kernel_size=3, stride=2 ),
              nn.Conv3d( in_channels=32, out_channels=64, kernel_size=3, stride=2 ) ]:
    # inplace operation
    conv1, num_pruned = pruning.prune_conv(conv, idxs=[0,1,2])
    print(conv,conv1, (conv1 is conv))
    # non-inplace operation
    conv2, num_pruned = pruning.prune_conv(conv, idxs=[0,1,2], inplace=False)
    print(conv, conv2, (conv2 is conv))
    # prune related layer
    conv3, num_pruned = pruning.prune_related_conv(conv, idxs=[0,1,2])
    print(conv3)

# Linear Layer
fc = nn.Linear(20, 30)
# inplace operation
fc1, num_pruned = pruning.prune_linear(fc, idxs=[0,1,2])
print(fc,fc1, (fc1 is fc))
# non-inplace operation
fc2, num_pruned = pruning.prune_linear(fc, idxs=[0,1,2], inplace=False)
print(fc, fc2, (fc2 is fc))
# prune related layer
fc3, num_pruned = pruning.prune_related_linear(fc, idxs=[0,1,2])
print(fc3)

# BatchNorm Layer
for bn in [ nn.BatchNorm1d(32),
            nn.BatchNorm2d(32),
            nn.BatchNorm3d(32) ]:
    # inplace operation
    bn1, num_pruned = pruning.prune_batchnorm(bn, idxs=[0,1,2])
    print(bn,bn1, (bn1 is bn))
    # non-inplace operation
    bn2, num_pruned = pruning.prune_batchnorm(bn, idxs=[0,1,2], inplace=False)
    print(bn, bn2, (bn2 is bn))

# PReLU Layer
prelu = nn.PReLU(32)
# inplace operation
prelu1, num_pruned = pruning.prune_prelu(prelu, idxs=[0,1,2])
print(prelu,prelu1, (prelu1 is prelu))
# non-inplace operation
prelu2, num_pruned = pruning.prune_prelu(prelu, idxs=[0,1,2], inplace=False)
print(prelu, prelu2, (prelu2 is prelu))
# no purning is prelu has only one parameter
prelu = nn.PReLU(1)
prelu1, num_pruned = pruning.prune_prelu(prelu, idxs=[0,1,2])
print(prelu,prelu1, (prelu1 is prelu))