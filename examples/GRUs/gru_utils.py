"""
Utilities for GRU pruning with torch-pruning library.

This module provides functionality to convert between PyTorch's built-in nn.GRU 
and custom PrunableGRU modules that are compatible with torch-pruning. The key 
innovation is breaking circular dependencies in recurrent layers by introducing 
identity layers that provide safe pruning points.

Key Components:
- PrunableGRU: Custom GRU implementation composed of exposed, prunable operators and identity layer for pruning
- Conversion functions between nn.GRU and PrunableGRU
- Model-wide replacement utilities for seamless integration
"""

import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class PrunableGRU(nn.Module):
    """
    Custom GRU module designed for compatibility with torch-pruning.
    
    This implementation replaces PyTorch's built-in nn.GRU with a custom version
    that includes identity layers (hidden_map) to break circular dependencies
    inherent in recurrent networks. This allows torch-pruning to safely modify
    hidden dimensions without encountering circular dependency constraints.
    
    Architecture:
    - Each layer contains: linear_ih, linear_hh, and hidden_map (identity layer)
    - The hidden_map provides a pruning point where dimensions can be safely modified
    - Maintains same hidden size across all layers (like torch.nn.GRU)
    
    Args:
        input_size (int): Number of expected input features
        hidden_size (int): Number of features in the hidden state
        num_layers (int, optional): Number of recurrent layers. Default: 1
        batch_first (bool, optional): If True, input/output tensors are provided 
            as (batch, seq, feature). Default: False
    
    Input Shape:
        - If batch_first=False: (seq_len, batch, input_size) or (seq_len, input_size)
        - If batch_first=True: (batch, seq_len, input_size)
        - hx: (num_layers, batch, hidden_size) or (num_layers, hidden_size)
    
    Output Shape:
        - output: Same shape as input but with input_size replaced by hidden_size
        - h_n: (num_layers, batch, hidden_size) or (num_layers, hidden_size)
    
    Note:
        This version maintains equal hidden sizes across layers for compatibility
        with torch.nn.GRU. For different hidden sizes per layer, use 
        PrunableGRUDifferentHiddenSize.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Create layers: each layer has linear_ih, linear_hh, hidden_map
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            layer_input_size = input_size if layer_idx == 0 else hidden_size
            gru_layer = nn.ModuleDict(
                {
                    "linear_ih": nn.Linear(layer_input_size, 3 * hidden_size),
                    "linear_hh": nn.Linear(hidden_size, 3 * hidden_size),
                    "hidden_map": nn.Linear(hidden_size, hidden_size),
                }
            )
            self.layers.append(gru_layer)

    def forward(self, x, hx=None):
        """
        Forward pass through the PrunableGRU.
        
        Implements the GRU equations with an additional identity mapping (hidden_map)
        that provides a safe pruning point for torch-pruning to modify hidden dimensions.
        
        GRU Equations:
        - r_t = σ(W_ir @ x_t + b_ir + W_hr @ h_(t-1) + b_hr)  # reset gate
        - z_t = σ(W_iz @ x_t + b_iz + W_hz @ h_(t-1) + b_hz)  # update gate  
        - n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_(t-1) + b_hn))  # new gate
        - h_t = (1 - z_t) * n_t + z_t * h_(t-1)  # new hidden state
        
        Args:
            x (torch.Tensor): Input tensor of shape:
                - (seq_len, input_size) for single batch, batch_first=False
                - (seq_len, batch_size, input_size) for batch, batch_first=False  
                - (batch_size, seq_len, input_size) for batch_first=True
            hx (torch.Tensor, optional): Initial hidden state tensor of shape:
                - (num_layers, hidden_size) for single batch
                - (num_layers, batch_size, hidden_size) for batch
                If None, defaults to zeros.
        
        Returns:
            tuple: (output, h_n) where:
                - output: Tensor containing output features for each timestep
                - h_n: Tensor containing final hidden state for each layer
        """
        batch_input = True
        if not self.batch_first:
            seq_len = x.shape[0]
            if len(x.shape) == 2:
                num_batch = 1
                x = x.unsqueeze(1)  # make it (seq_len, batch, input_size)
                if hx is not None:
                    hx = hx.unsqueeze(1)
                batch_input = False
            else:
                num_batch = x.shape[1]
        else:
            seq_len = x.shape[1]
            num_batch = x.shape[0]
            x = x.permute(1, 0, 2)

        # hidden state initialization
        if hx is None:
            hx = torch.zeros(self.num_layers, num_batch, self.hidden_size, device=x.device)

        # hidden state output
        h_n = []

        # out tracks the output of last layer, which is input to the next layer
        out = x

        for layer_idx, layer in enumerate(self.layers):
            h_prev = hx[layer_idx, 0, :].unsqueeze(0)  # (batch, hidden_size)
            outputs = []  # to contain the outputs for each time step for this layer
            for t in range(seq_len):

                h = layer["hidden_map"](h_prev)
                gates_hh = layer["linear_hh"](h)
                gates_ih = layer["linear_ih"](out[t, 0, :].unsqueeze(0))

                r_hh_lin_out, z_hh_lin_out, n_hh_lin_out = gates_hh.chunk(3, dim=1)
                r_ih_lin_out, z_ih_lin_out, n_ih_lin_out = gates_ih.chunk(3, dim=1)

                r = torch.sigmoid(r_hh_lin_out + r_ih_lin_out)
                z = torch.sigmoid(z_hh_lin_out + z_ih_lin_out)
                n = torch.tanh(n_ih_lin_out + r * n_hh_lin_out)

                h_new = (1 - z) * n + z * h

                if layer_idx > 0:
                    # this forces that the hidden state size to be the same across all layers
                    # will reduce pruning flexibility but is required for torch.nn.GRU compatibility when num_layers > 1
                    outputs.append(h_new + 0 * out[t, 0, :].unsqueeze(0))
                else:
                    outputs.append(h_new)

                h_prev = h_new

            out = torch.stack(outputs, dim=0)  # (seq_len, batch, hidden_size)
            h_n.append(h_prev)  # keep track of final hidden state output for each layer

        # Stack hidden states for all layers: (num_layers, batch, hidden_size)
        h_n = torch.stack(h_n, dim=0)
        if not batch_input:
            h_n = h_n.squeeze(1)
            out = out.squeeze(1)  # (seq_len, hidden_size)

        if self.batch_first:
            out = out.permute(1, 0, 2)

        return out, h_n


def torchgru_to_prunablegru(gru):
    """
    Converts a standard nn.GRU layer to a PrunableGRU with identical behavior.
    
    This function creates a PrunableGRU instance and copies all weights and biases
    from the original nn.GRU. The hidden_map layers are initialized as identity
    transformations to ensure mathematical equivalence before pruning.
    
    Args:
        gru (nn.GRU): A PyTorch GRU layer to convert
        
    Returns:
        PrunableGRU: Equivalent PrunableGRU with copied weights and identity mappings
        
    Example:
        >>> original_gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2)
        >>> prunable_gru = torchgru_to_prunablegru(original_gru)
        >>> # prunable_gru now behaves identically to original_gru
    """
    input_size = gru.input_size
    hidden_size = gru.hidden_size
    num_layers = gru.num_layers
    batch_first = gru.batch_first

    prunable_gru = PrunableGRU(
        input_size, hidden_size, num_layers=num_layers, batch_first=batch_first
    )

    with torch.no_grad():
        for i in range(num_layers):
            layer = prunable_gru.layers[i]
            layer["linear_ih"].weight.copy_(getattr(gru, f"weight_ih_l{i}"))
            layer["linear_ih"].bias.copy_(getattr(gru, f"bias_ih_l{i}"))
            layer["linear_hh"].weight.copy_(getattr(gru, f"weight_hh_l{i}"))
            layer["linear_hh"].bias.copy_(getattr(gru, f"bias_hh_l{i}"))

            # hidden_map init as identity matrix + zero bias
            layer["hidden_map"].weight.copy_(torch.eye(hidden_size))
            layer["hidden_map"].bias.zero_()

    return prunable_gru


def replace_torchgru_with_prunablegru(original_model):
    """
    Creates a deep copy of a model with all nn.GRU modules replaced by PrunableGRU.
    
    This function recursively traverses the model architecture and replaces every
    nn.GRU instance with an equivalent PrunableGRU. This is the recommended way
    to prepare a model for GRU pruning with torch-pruning.
    
    The conversion process:
    1. Creates a deep copy of the original model
    2. Recursively finds all nn.GRU modules
    3. Replaces each with a PrunableGRU using torchgru_to_prunablegru()
    4. Preserves the original model's device placement
    
    Args:
        original_model (nn.Module): PyTorch model containing nn.GRU layers
        
    Returns:
        nn.Module: Deep copy of the model with PrunableGRU layers, preserving
                  all other components and maintaining device placement
                  
    Example:
        >>> model = MyModel()  # Contains nn.GRU layers
        >>> prunable_model = replace_torchgru_with_prunablegru(model)
        >>> # Now ready for pruning with torch-pruning
    """
    model_copy = copy.deepcopy(original_model)

    # Get the device of the original model
    device = next(original_model.parameters()).device

    def _replace_gru(module):
        for name, child in module.named_children():
            if isinstance(child, nn.GRU):
                prunable_gru = torchgru_to_prunablegru(child)
                setattr(module, name, prunable_gru)
            else:
                _replace_gru(child)

    _replace_gru(model_copy)

    # Move the copied model to the same device as the original model
    model_copy.to(device)
    return model_copy


def prunablegru_to_torchgru(prunable_gru):
    """
    Converts a PrunableGRU back to a standard nn.GRU layer.
    
    This function is typically used after pruning to convert the pruned PrunableGRU
    back to a standard nn.GRU for deployment. The conversion discards the identity
    hidden_map layers while preserving the pruned structure and learned weights.
    
    Important: The resulting nn.GRU will have dimensions matching the pruned
    PrunableGRU, not the original pre-pruning dimensions. This allows you to
    deploy pruned models using standard PyTorch components.
    
    Args:
        prunable_gru (PrunableGRU): A PrunableGRU instance (potentially pruned)
        
    Returns:
        nn.GRU: Standard PyTorch GRU with copied weights and biases.
               Dimensions match the (potentially pruned) PrunableGRU.
               
    Example:
        >>> # After pruning
        >>> pruned_prunable_gru = prune_model(prunable_gru)
        >>> standard_gru = prunablegru_to_torchgru(pruned_prunable_gru) 
        >>> # standard_gru now has pruned dimensions but uses nn.GRU
    """
    # Get parameters
    num_layers = prunable_gru.num_layers
    hidden_size = prunable_gru.layers[0]["linear_hh"].weight.shape[1]
    # The input size is from first layer's linear_ih input features
    input_size = prunable_gru.layers[0]["linear_ih"].weight.shape[1]

    # Create torch GRU with matching parameters
    torch_gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    with torch.no_grad():
        for i in range(num_layers):
            layer = prunable_gru.layers[i]
            # Copy weights and biases properly:
            getattr(torch_gru, f"weight_ih_l{i}").data.copy_(layer["linear_ih"].weight.data)
            getattr(torch_gru, f"bias_ih_l{i}").data.copy_(layer["linear_ih"].bias.data)
            getattr(torch_gru, f"weight_hh_l{i}").data.copy_(layer["linear_hh"].weight.data)
            getattr(torch_gru, f"bias_hh_l{i}").data.copy_(layer["linear_hh"].bias.data)

    return torch_gru


def replace_prunablegru_with_torchgru(original_model):
    """
    Creates a deep copy of a model with all PrunableGRU modules replaced by nn.GRU.
    
    This function is the inverse of replace_torchgru_with_prunablegru() and is
    typically used after pruning to convert back to standard PyTorch components
    for deployment. The resulting model maintains the pruned structure but uses
    standard nn.GRU layers for better compatibility and potentially improved performance.
    
    The conversion process:
    1. Creates a deep copy of the model containing PrunableGRU layers
    2. Recursively finds all PrunableGRU modules  
    3. Replaces each with an nn.GRU using prunablegru_to_torchgru()
    4. Preserves device placement and all other model components
    
    Args:
        original_model (nn.Module): PyTorch model containing PrunableGRU layers
        
    Returns:
        nn.Module: Deep copy of the model with standard nn.GRU layers.
                  Dimensions reflect any pruning that was applied to the
                  PrunableGRU layers.
                  
    Example:
        >>> # After pruning a model with PrunableGRU layers
        >>> pruned_model = prune_with_torch_pruning(model_with_prunable_gru)
        >>> final_model = replace_prunablegru_with_torchgru(pruned_model)
        >>> # final_model now uses standard nn.GRU with pruned dimensions
    """
    model_copy = copy.deepcopy(original_model)

    # Get the device of the original model
    device = next(original_model.parameters()).device

    def _replace_prunablegru(module):
        for name, child in module.named_children():
            # Check for multilayer prunable GRU
            if isinstance(child, PrunableGRU):

                torch_gru = prunablegru_to_torchgru(child)

                setattr(module, name, torch_gru)
            else:
                _replace_prunablegru(child)

    _replace_prunablegru(model_copy)
    # Move the copied model to the same device as the original model
    model_copy.to(device)
    return model_copy

class GRUTestNet(torch.nn.Module):
    """
    Simple test network demonstrating GRU pruning workflow.
    
    Architecture: Conv layers → FC layers → Multi-layer GRU → Output FC
    This mimics common architectures where GRU processes encoded features.
    """
    def __init__(self, input_size=80, hidden_size=164):
        super(GRUTestNet, self).__init__()
        # Feature extraction layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 196)
        self.fc2 = nn.Linear(196, 80)
        
        # Multi-layer GRU (this is what we want to prune)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1)
        
        # Output layer
        self.fc3 = nn.Linear(164, 10)

    def forward(self, x, hx=None):
        # Feature extraction
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # GRU processing (sequence length = 1 for this example)
        x = self.gru(x, hx=hx)[0]

        # Final classification
        x = self.fc3(x)
        return x