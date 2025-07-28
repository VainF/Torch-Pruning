# GRU Pruning with Torch-Pruning

This example demonstrates how to prune GRU (Gated Recurrent Unit) layers in PyTorch models using the torch-pruning library. The key challenge addressed here is making GRU layers compatible with torch-pruning through a prunable GRU implementation.

## Key Innovations

### Problem 1: Opaque C++ Implementation
This example demonstrates how to prune GRU layers using torch-pruning. Since `torch.nn.GRU` uses an opaque C++ implementation that torch-pruning cannot analyze, we provide a `PrunableGRU` implementation that exposes internal operations for structural pruning.

### Problem 2: Circular Dependency in Hidden States
GRU layers create circular dependencies that prevent torch-pruning from properly modifying hidden dimensions. In other words, torch-pruning sees the hidden state as both a module input AND output and torch-pruning does not naturally have the freedom to update the size of model inputs. 

## Solution: Custom PrunableGRU with Hidden State Identity "Buffer" Layer

Our approach addresses both problems:

1. **Replace opaque torch.nn.GRU** with a custom `PrunableGRU` implementation that exposes all internal operations as standard PyTorch layers that are prunable by torch-pruning. These modules can be replaced by PyTorch GRU after pruning is performed.
2. **Identity layers (`hidden_map`)** in `PrunableGRU` to break the circular dependency:

hidden_state → GRU → hidden_map (identity) → pruned_hidden_state

This provides "safe" pruning points where torch-pruning can modify dimensions without worrying about the circular constraint.

## Workflow

1. **Convert**: Replace `torch.nn.GRU` with `PrunableGRU` (includes identity layers)
2. **Prune**: Run torch-pruning (can now safely modify hidden dimensions)  
3. **Convert Back**: Convert back to `torch.nn.GRU` (removes identity layers, keeps pruned structure)

## Usage

A basic example is in `gru_pruning_example.py`. 

Run the example:
```bash
python gru_pruning_example.py
```

**What this example *does* include**: A demonstration of the mechanics of pruning GRUs in a very simple test network. In particular, we demonstrate that the GRU hidden and input sizes are smaller after pruning is performed and that the model can still perform inference after pruning.

**What this example *does not* include**: An analysis of  performance in a useful model after GRU pruning is performed. We leave this to the user to explore.

## Files
- `gru_pruning_example.py` - Complete working example
- `gru_utils.py` - PrunableGRU implementation and utilities
- `test_gru.py` - Unit tests of the gru pruning utilities in `gru_utils.py`
- `README.md` - This file

## Limitations

- Sequence length: Supports sequence_length=1 only during pruning; in torch-pruning `example_inputs` argument must correspond to sequence_length=1.
- Batch size: Supports batch_size=1 only during pruning; if gru input data has a batch dimension, torch-pruning `example_inputs` argument must correspond to batch_size=1.
- Multi-layer: Tested with single-layer GRUs only. We recommend that multilayer `torch.nn.GRU`s are "unwrapped" prior to pruning, into multiple cascaded single layer `torch.nn.GRU`s, to allow pruning to achieve different hidden state sizes across layers

## Extensions

The approach demonstrated here can be extended to:
- LSTM layers
- minGRU/minLSTM architectures
- Other recurrent architectures with similar circular dependency issues that can be decomposed into prunable building blocks

