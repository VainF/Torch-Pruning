"""
This script demonstrates the pruning of GRU modules in a PyTorch model and tests out some of the different building blocks.
This work was a precurser to testing GRU pruning in our actual DeepFilterNet model.
"""
import torch
import torch.nn as nn
import torch_pruning as tp
from gru_utils import (
    replace_prunablegru_with_torchgru,
    replace_torchgru_with_prunablegru,
    torchgru_to_prunablegru,
    GRUTestNet,
)


class TestPrunableGRUBehavior:
    """Test that prunable GRU modules behave identically to torch.nn.GRU."""
    def gru_params(self):
        return {
            "input_size": 80,
            "hidden_size": 164,
            "num_layers": 1
        }
        
    def test_single_layer_gru_equivalence(self):
        """Test single layer GRU equivalence."""
        # get params for single layer GRU
        params = self.gru_params()
        
        # create torch GRU and prunable GRU
        torch_gru = nn.GRU(**params)
        
        # copy weights from torch GRU to prunable GRU
        prunable_gru = torchgru_to_prunablegru(torch_gru)
        
        # run same input through both units
        seq_length = 2
        gru_input = torch.randn([seq_length, params["input_size"]]) # test sequence length of 2
        gru_state = torch.randn([params["num_layers"], params["hidden_size"]])
        
        torch_gru_out = torch_gru(gru_input, hx=gru_state)[0]
        prunable_gru_out = prunable_gru(gru_input, hx=gru_state)[0]
        
        assert torch.allclose(
            torch_gru_out, prunable_gru_out, atol=1e-6
        ), "Single layer GRU outputs do not match!"
        
    def test_multi_layer_gru_equivalence(self):
        """Test multi layer GRU equivalence."""
        # get params for multi-layer GRU
        params = self.gru_params()
        params["num_layers"] = 2  # test multiple layers
        
        # create torch GRU and prunable GRU
        torch_gru = nn.GRU(**params)
        
        # copy weights from torch GRU to prunable GRU
        prunable_gru = torchgru_to_prunablegru(torch_gru)
        
        # run same input through both units
        seq_length = 2
        gru_input = torch.randn([seq_length, params["input_size"]]) # test sequence length of 2
        gru_state = torch.randn([params["num_layers"], params["hidden_size"]])
        
        torch_gru_out = torch_gru(gru_input, hx=gru_state)[0]
        prunable_gru_out = prunable_gru(gru_input, hx=gru_state)[0]
        
        assert torch.allclose(
            torch_gru_out, prunable_gru_out, atol=1e-6
        ), "Multi layer GRU outputs do not match!"
        
    def test_gru_equivalence_with_batch_dim_input(self):
        """Test GRU equivalence with batch dimension input. (only can have batch size = 1 for now)"""
        params = self.gru_params()
        
        # create torch GRU and prunable GRU
        torch_gru = nn.GRU(**params)
        
        # copy weights from torch GRU to prunable GRU
        prunable_gru = torchgru_to_prunablegru(torch_gru)
        
        # run same input through both units
        batch_size = 1
        seq_len = 2
        gru_input = torch.randn([seq_len, batch_size, params["input_size"]]) # test sequence length of 2
        gru_state = torch.randn([params["num_layers"], batch_size, params["hidden_size"]])
        torch_gru_out = torch_gru(gru_input, hx=gru_state)[0]
        prunable_gru_out = prunable_gru(gru_input, hx=gru_state)[0]
        
        assert torch.allclose(
            torch_gru_out, prunable_gru_out, atol=1e-6
        ), "GRU outputs match with batch dim input!"
    
    def test_gru_equivalence_with_batch_first(self):
        params = self.gru_params()
        
        batch_size = 1
        seq_len = 2
        
        # try out batch_first=True, to match DFN network
        gru_input = torch.randn([batch_size, seq_len, 80]) # batch, seq, input len
        gru_state = torch.randn([params["num_layers"], batch_size, params["hidden_size"]])
        
        # create torch GRU and prunable GRU
        torch_gru = nn.GRU(input_size=params["input_size"], hidden_size=params["hidden_size"], num_layers=params["num_layers"], batch_first=True)
        prunable_gru = torchgru_to_prunablegru(torch_gru)
        
        # pass same input through both units
        torch_gru_out = torch_gru(gru_input, hx=gru_state)[0]
        prunable_gru_out = prunable_gru(gru_input, hx=gru_state)[0]
        
        # verify outputs match
        assert torch.allclose(
            torch_gru_out, prunable_gru_out, atol=1e-6
        ), "Batch-first GRU outputs do not match!"   

class TestGRUPruneUtils:
    """Test utility functions for GRU pruning."""
    
    # here we are testing the prunable gru where hidden size has to be the same across layers!
    def get_model_input(self):
        return torch.randn(1, 1, 28, 28)
    
    def test_replacement_utils(self):
        model_input = self.get_model_input()
        model_torchGRU = GRUTestNet()
        
        # test process of finding GRUs and replacing with prunable GRU. For use pre-pruning.
        model_prunableGRU = replace_torchgru_with_prunablegru(model_torchGRU)
        model_torchGRU_out = model_torchGRU(model_input)
        model_prunableGRU_out = model_prunableGRU(model_input)
        assert torch.allclose(
            model_torchGRU_out, model_prunableGRU_out, atol=1e-6
        ), "Outputs of original and prunable GRU models do not match!"
        
        # test process of going the other way, for post-pruning
        model_torchGRU_copy = replace_prunablegru_with_torchgru(model_prunableGRU)
        model_torchGRU_out = model_torchGRU_copy(model_input)
        assert torch.allclose(
            model_torchGRU_out, model_prunableGRU_out, atol=1e-6
        ), "Outputs of original and prunable GRU models do not match after conversion back!"


if __name__ == "__main__":
    test_behavior = TestPrunableGRUBehavior()
    test_behavior.test_single_layer_gru_equivalence()
    test_behavior.test_multi_layer_gru_equivalence()
    test_behavior.test_gru_equivalence_with_batch_dim_input()
    test_behavior.test_gru_equivalence_with_batch_first()
    
    test_replacement_utils = TestGRUPruneUtils()
    test_replacement_utils.test_replacement_utils()
    print("All tests passed!")
