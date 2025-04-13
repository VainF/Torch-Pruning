import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import vit_b_16 as entry
import torch_pruning as tp
from torchvision.models.vision_transformer import VisionTransformer

def test_serialization():
    model = entry().eval()

    customized_value = 8
    model.customized_value = customized_value
    importance = tp.importance.MagnitudeImportance(p=1)
    round_to = None
    if isinstance( model, VisionTransformer): round_to = model.encoder.layers[0].num_heads
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=torch.randn(1, 3, 224, 224),
        importance=importance,
        iterative_steps=1,
        pruning_ratio=0.5,
        round_to=round_to,
    )
    pruner.step()
    if isinstance(
        model, VisionTransformer
    ):  # Torchvision relies on the hidden_dim variable for forwarding, so we have to modify this varaible after pruning
        model.hidden_dim = model.conv_proj.out_channels
        true_hidden_dim = model.hidden_dim
        print(model.class_token.shape, model.encoder.pos_embedding.shape)
    
    state_dict = tp.state_dict(model)
    torch.save(state_dict, 'test.pth')

    # create a new model
    model = entry().eval()
    print(model)

    # load the pruned state_dict
    loaded_state_dict = torch.load('test.pth', map_location='cpu', weights_only=False)
    tp.load_state_dict(model, state_dict=loaded_state_dict)
    print(model)

    # test
    assert model.customized_value == customized_value
    assert model.hidden_dim == true_hidden_dim
    print(model.customized_value) # check the user attributes
    print(model.hidden_dim)

    out = model(torch.randn(1,3,224,224))
    print(out.shape)
    loss = out.sum()
    loss.backward()

if __name__=='__main__':
    test_serialization()