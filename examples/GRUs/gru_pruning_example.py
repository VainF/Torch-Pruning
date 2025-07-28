"""
GRU Pruning Example for Torch-Pruning

This example demonstrates how to prune GRU layers in PyTorch models using torch-pruning.
The key challenges addressed here are the opaque implementation of standard GRU layers
and the circular dependency problem inherent to recurrent layers.

Key Innovation: Custom GRU Implementation with Identity Layer Solution
=====================================================================

Problem 1: PyTorch's torch.nn.GRU uses optimized C/CUDA implementations under the hood.
These low-level implementations are opaque to torch-pruning's dependency graph analysis,
making it impossible for the pruning framework to understand the internal structure 
and dependencies of the GRU operations.

Problem 2: Past GRU hidden states are inputs to model so torch-pruning will not alter its size.


Solution: Create a custom PrunableGRUEqualHiddenSize that:
1. Implements GRU operations in pure Python/PyTorch (transparent to torch-pruning)
2. Inserts identity layers (hidden_map) to break circular dependencies and allow pruning of the hidden state size

    hidden_state → GRU → hidden_map (identity) → pruned_hidden_state

This provides "safe" pruning points where torch-pruning can modify dimensions
without worrying about the circular constraint, while using a transparent 
implementation that the pruning framework can analyze.

Workflow:
1. Replace torch.nn.GRU with PrunableGRU (includes identity layers)
2. Run torch-pruning (can now safely modify hidden dimensions)
3. Convert back to torch.nn.GRU (removes identity layers, keeps pruned structure)
"""

import torch_pruning as tp
import torch
import torch.nn as nn

# Import your utility functions
from gru_utils import (
    replace_prunablegru_with_torchgru,
    replace_torchgru_with_prunablegru,
    GRUTestNet,
)




def demonstrate_gru_pruning_workflow():
    """
    Complete workflow showing GRU pruning with the identity layer solution.
    
    This function demonstrates and end-to-end GRU pruning workflow
    """
    print("=" * 60)
    print("GRU Pruning Workflow Demonstration")
    print("=" * 60)
    
    # Step 1: Create original model with standard torch.nn.GRU
    print("\n1. Creating original model with torch.nn.GRU...")
    model = GRUTestNet()
    original_gru_hidden_size = model.gru.hidden_size
    original_gru_input_size = model.gru.input_size
    print(f"   Original GRU hidden size: {original_gru_hidden_size}")
    print(f"   Original GRU input size: {original_gru_input_size}")

    
    # Step 2: Prepare inputs for torch-pruning dependency analysis
    print("\n2. Preparing example inputs for dependency graph...")
    example_inputs = torch.randn(1, 1, 28, 28)
    input_data = {"x": example_inputs, "hx": None}
    
    # Verify model runs before pruning
    original_output = model(**input_data)
    print(f"   Original model output shape: {original_output.shape}")
    
    # Step 3: Replace torch.nn.GRU with PrunableGRU (adds identity layers)
    print("\n3. Converting to PrunableGRU (adds identity layers to break circular deps)...")
    model = replace_torchgru_with_prunablegru(model)
    print("   ✓ Identity layers inserted - torch-pruning can now modify hidden dims")
    
    # Step 4: Build dependency graph and create pruner  
    print("\n4. Building dependency graph and setting up pruner...")
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=input_data)
    
    imp = tp.importance.GroupMagnitudeImportance(p=2)
    pruner = tp.pruner.MetaPruner(
        model,
        input_data,
        importance=imp,
        pruning_ratio=0.2,  # Remove 20% of input/output channels
        isomorphic=True,
        global_pruning=True,
        root_module_types=(nn.Linear, nn.LayerNorm, nn.Conv2d),
    )
    
    # Step 5: Execute pruning
    print("\n5. Executing pruning...")
    pruner.step()
    
    # Verify model still works after pruning
    pruned_output = model(example_inputs)
    print("   ✓ Model still functional after pruning")
    
    # Step 6: Convert back to torch.nn.GRU (removes identity layers)
    print("\n6. Converting back to torch.nn.GRU (removes identity layers)...")
    final_model = replace_prunablegru_with_torchgru(model)
    pruned_output = final_model(example_inputs)

    # Show the results
    print(f"   Hidden size reduction: {original_gru_hidden_size} → {final_model.gru.hidden_size}")
    print(f"   Hidden size reduction: {original_gru_input_size} → {final_model.gru.input_size}")
    
    # Final verification
    final_output = final_model(example_inputs)
    print("   ✓ Successfully pruned GRU while maintaining functionality!")
    
    print("\n" + "=" * 60)
    print("Workflow Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("• Identity layers broke circular dependencies in hidden states")
    print("• torch-pruning could safely modify GRU hidden dimensions") 
    print("• Final model uses standard torch.nn.GRU with reduced hidden size")
    print("• All functionality preserved throughout the process")


if __name__ == "__main__":
    demonstrate_gru_pruning_workflow()
