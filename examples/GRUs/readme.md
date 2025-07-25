# GRU Pruning with Torch-Pruning

This example demonstrates how to prune GRU (Gated Recurrent Unit) layers in PyTorch models using the torch-pruning library. The key challenge addressed here is making GRU layers compatible with torch-pruning through custom implementations that resolve fundamental architectural constraints.

## Key Innovations

### Problem 1: Opaque C++ Implementation
The standard `torch.nn.GRU` module uses an optimized C++ implementation under the hood that torch-pruning cannot analyze or modify. This black-box nature prevents the pruning library from understanding the internal structure needed for safe pruning operations.

### Problem 2: Circular Dependency in Hidden States
GRU layers create circular dependencies that prevent torch-pruning from modifying hidden dimensions:

torch-pruning sees the hidden state as both input AND output, so it refuses to change the hidden dimension to avoid breaking this cycle.

## Solution: Custom PrunableGRU with Identity Layers

Our approach addresses both problems:

1. **Replace opaque torch.nn.GRU** with a custom `PrunableGRU` implementation that exposes all internal operations as standard PyTorch layers
2. **Insert identity layers (`hidden_map`)** to break the circular dependency:

hidden_state → GRU → hidden_map (identity) → pruned_hidden_state


This provides "safe" pruning points where torch-pruning can modify dimensions without worrying about the circular constraint.

## Workflow

1. **Convert**: Replace `torch.nn.GRU` with `PrunableGRU` (includes identity layers)
2. **Prune**: Run torch-pruning (can now safely modify hidden dimensions)  
3. **Convert Back**: Convert back to `torch.nn.GRU` (removes identity layers, keeps pruned structure)

## Usage

### Basic Example

```python
import torch
import torch.nn as nn
from gru_utils import replace_torchgru_with_prunablegru, replace_prunablegru_with_torchgru


#### Original model with torch.nn.GRU
