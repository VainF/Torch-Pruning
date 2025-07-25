"""
Comprehensive tests for GRU pruning functionality.
Tests both the prunable GRU implementation and the pruning workflow.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp

from df.torch_prune_utils import (
    replace_prunablegru_with_torchgru_equalhidden,
    replace_torchgru_with_prunablegru_equalhidden,
    torchgru_to_prunablegru_equalhidden,
)


class GRUTestNet(torch.nn.Module):
    """Simple test network for GRU pruning (seq_first=True)."""
    def __init__(self, input_size=80, hidden_size=164):
        super(GRUTestNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 196)
        self.fc2 = nn.Linear(196, 80)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2)
        self.fc3 = nn.Linear(164, 10)

    def forward(self, x, hx=None):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.gru(x, hx=hx)[0]
        x = self.fc3(x)
        return x


class GRUTestNetBatchFirst(torch.nn.Module):
    """Simple test network for GRU pruning (batch_first=True)."""
    def __init__(self, input_size=80, hidden_size=164):
        super(GRUTestNetBatchFirst, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 196)
        self.fc2 = nn.Linear(196, 80)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc3 = nn.Linear(164, 10)

    def forward(self, x, hx=None):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.gru(x, hx=hx)[0]
        x = x.squeeze(1)  # Remove sequence dimension
        x = self.fc3(x)
        return x


class TestPrunableGRUBehavior:
    """Test that prunable GRU modules behave identically to torch.nn.GRU."""
    
    @pytest.fixture
    def gru_params(self):
        return {
            "input_size": 80,
            "hidden_size": 164,
            "num_layers": 2
        }
    
    def test_single_layer_gru_equivalence(self, gru_params):
        """Test single layer GRU equivalence."""
        params = gru_params.copy()
        params["num_layers"] = 1
        
        torch_gru = nn.GRU(**params)
        prunable_gru = torchgru_to_prunablegru_equalhidden(torch_gru)
        
        # Test with sequence-first input (default)
        gru_input = torch.randn([3, params["input_size"]])  # seq_len=3
        gru_state = torch.randn([1, params["hidden_size"]])
        
        torch_gru_out = torch_gru(gru_input, hx=gru_state)[0]
        prunable_gru_out = prunable_gru(gru_input, hx=gru_state)[0]
        
        assert torch.allclose(torch_gru_out, prunable_gru_out, atol=1e-6), \
            "Single-layer GRU outputs do not match!"
    
    def test_multi_layer_gru_equivalence(self, gru_params):
        """Test multi-layer GRU equivalence."""
        torch_gru = nn.GRU(**gru_params)
        prunable_gru = torchgru_to_prunablegru_equalhidden(torch_gru)
        
        # Test with sequence-first input (default)
        gru_input = torch.randn([2, gru_params["input_size"]])  # seq_len=2
        gru_state = torch.randn([gru_params["num_layers"], gru_params["hidden_size"]])
        
        torch_gru_out = torch_gru(gru_input, hx=gru_state)[0]
        prunable_gru_out = prunable_gru(gru_input, hx=gru_state)[0]
        
        assert torch.allclose(torch_gru_out, prunable_gru_out, atol=1e-6), \
            "Multi-layer GRU outputs do not match!"
    
    def test_batch_first_equivalence(self, gru_params):
        """Test batch_first=True equivalence."""
        batch_size = 1
        seq_len = 2
        
        torch_gru = nn.GRU(**gru_params, batch_first=True)
        prunable_gru = torchgru_to_prunablegru_equalhidden(torch_gru)
        
        # Test with batch-first input
        gru_input = torch.randn([batch_size, seq_len, gru_params["input_size"]])
        gru_state = torch.randn([gru_params["num_layers"], batch_size, gru_params["hidden_size"]])
        
        torch_gru_out = torch_gru(gru_input, hx=gru_state)[0]
        prunable_gru_out = prunable_gru(gru_input, hx=gru_state)[0]
        
        assert torch.allclose(torch_gru_out, prunable_gru_out, atol=1e-6), \
            "Batch-first GRU outputs do not match!"
    
    def test_no_hidden_state_equivalence(self, gru_params):
        """Test equivalence when no hidden state is provided."""
        torch_gru = nn.GRU(**gru_params)
        prunable_gru = torchgru_to_prunablegru_equalhidden(torch_gru)
        
        gru_input = torch.randn([2, gru_params["input_size"]])
        
        torch_gru_out = torch_gru(gru_input)[0]
        prunable_gru_out = prunable_gru(gru_input)[0]
        
        assert torch.allclose(torch_gru_out, prunable_gru_out, atol=1e-6), \
            "GRU outputs without hidden state do not match!"


class TestModelConversion:
    """Test model-level conversion between torch and prunable GRU implementations."""
    
    def test_model_conversion_seq_first(self):
        """Test model conversion with seq_first=True (default)."""
        original_model = GRUTestNet()
        example_input = torch.randn(1, 1, 28, 28)
        
        # Get original output
        original_output = original_model(example_input)
        
        # Convert to prunable GRU
        prunable_model = replace_torchgru_with_prunablegru_equalhidden(original_model)
        prunable_output = prunable_model(example_input)
        
        assert torch.allclose(original_output, prunable_output, atol=1e-6), \
            "Model outputs do not match after conversion to prunable GRU!"
        
        # Convert back to torch GRU
        converted_back_model = replace_prunablegru_with_torchgru_equalhidden(prunable_model)
        converted_back_output = converted_back_model(example_input)
        
        assert torch.allclose(prunable_output, converted_back_output, atol=1e-6), \
            "Model outputs do not match after conversion back to torch GRU!"
    
    def test_model_conversion_batch_first(self):
        """Test model conversion with batch_first=True."""
        original_model = GRUTestNetBatchFirst()
        example_input = torch.randn(1, 1, 28, 28)
        
        # Get original output
        original_output = original_model(example_input)
        
        # Convert to prunable GRU
        prunable_model = replace_torchgru_with_prunablegru_equalhidden(original_model)
        prunable_output = prunable_model(example_input)
        
        assert torch.allclose(original_output, prunable_output, atol=1e-6), \
            "Batch-first model outputs do not match after conversion to prunable GRU!"
        
        # Convert back to torch GRU
        converted_back_model = replace_prunablegru_with_torchgru_equalhidden(prunable_model)
        converted_back_output = converted_back_model(example_input)
        
        assert torch.allclose(prunable_output, converted_back_output, atol=1e-6), \
            "Batch-first model outputs do not match after conversion back to torch GRU!"


class TestPruningWorkflow:
    """Test the complete pruning workflow."""
    
    @pytest.mark.parametrize("batch_first", [True, False])
    def test_pruning_workflow(self, batch_first):
        """Test complete pruning workflow."""
        # Create model
        if batch_first:
            model = GRUTestNetBatchFirst()
        else:
            model = GRUTestNet()
        
        example_input = torch.randn(1, 1, 28, 28)
        input_data = {"x": example_input, "hx": None}
        
        # Get original output size for comparison
        original_output = model(**input_data)
        original_gru_hidden_size = model.gru.hidden_size
        
        # Convert to prunable GRU
        model = replace_torchgru_with_prunablegru_equalhidden(model)
        
        # Build dependency graph
        DG = tp.DependencyGraph().build_dependency(model, example_inputs=input_data)
        
        # Set up pruning
        imp = tp.importance.GroupMagnitudeImportance(p=2)
        pruning_ratio = 0.5
        pruner = tp.pruner.MetaPruner(
            model,
            input_data,
            importance=imp,
            pruning_ratio=pruning_ratio,
            isomorphic=True,
            global_pruning=True,
            root_module_types=(nn.Linear, nn.LayerNorm, nn.Conv2d),
        )
        
        # Execute pruning
        pruner.step()
        
        # Test inference after pruning
        pruned_output = model(**input_data)
        assert pruned_output.shape == original_output.shape, \
            "Output shape changed after pruning!"
        
        # Convert back to torch GRU
        final_model = replace_prunablegru_with_torchgru_equalhidden(model)
        
        # Test final inference
        final_output = final_model(**input_data)
        assert final_output.shape == original_output.shape, \
            "Final output shape changed after conversion!"
        
        # Verify that hidden size was actually reduced
        final_gru_hidden_size = final_model.gru.hidden_size
        assert final_gru_hidden_size < original_gru_hidden_size, \
            f"Hidden size was not reduced: {final_gru_hidden_size} >= {original_gru_hidden_size}"
        
        print(f"Original hidden size: {original_gru_hidden_size}")
        print(f"Pruned hidden size: {final_gru_hidden_size}")
        print(f"Reduction ratio: {final_gru_hidden_size / original_gru_hidden_size:.2f}")
    
    def test_pruning_preserves_functionality(self):
        """Test that pruned model maintains basic functionality."""
        model = GRUTestNet()
        example_input = torch.randn(1, 1, 28, 28)
        input_data = {"x": example_input, "hx": None}
        
        # Convert and prune
        model = replace_torchgru_with_prunablegru_equalhidden(model)
        
        imp = tp.importance.GroupMagnitudeImportance(p=2)
        pruner = tp.pruner.MetaPruner(
            model,
            input_data,
            importance=imp,
            pruning_ratio=0.3,  # Moderate pruning
            global_pruning=True,
        )
        pruner.step()
        
        # Convert back
        final_model = replace_prunablegru_with_torchgru_equalhidden(model)
        
        # Test with different inputs
        test_inputs = [
            torch.randn(1, 1, 28, 28),
            torch.randn(2, 1, 28, 28),  # Different batch size
        ]
        
        for test_input in test_inputs:
            try:
                output = final_model(test_input, hx=None)
                assert output.shape[0] == test_input.shape[0], \
                    f"Batch dimension mismatch: {output.shape[0]} != {test_input.shape[0]}"
                assert torch.isfinite(output).all(), \
                    "Model output contains non-finite values!"
            except Exception as e:
                pytest.fail(f"Model failed on input shape {test_input.shape}: {e}")


if __name__ == "__main__":
    # Run basic functionality test
    test_behavior = TestPrunableGRUBehavior()
    test_behavior.test_multi_layer_gru_equivalence({
        "input_size": 80,
        "hidden_size": 164,
        "num_layers": 2
    })
    
    test_conversion = TestModelConversion()
    test_conversion.test_model_conversion_seq_first()
    test_conversion.test_model_conversion_batch_first()
    
    test_pruning = TestPruningWorkflow()
    test_pruning.test_pruning_workflow(batch_first=True)
    test_pruning.test_pruning_workflow(batch_first=False)
    
    print("All tests passed!")