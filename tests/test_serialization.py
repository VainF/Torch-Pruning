import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import densenet121
import torch_pruning as tp

def test_depgraph():
    model = densenet121().eval()

    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))

    for group in DG.get_all_groups():
        # handle groups in sequential order
        idxs = [2,4,6] # my pruning indices
        group.prune(idxs=idxs)

    state_dict = tp.state_dict(model)
    torch.save(state_dict, 'test.pth')

    # create a new model
    
    model = densenet121().eval()
    print(model)

    # load the pruned state_dict
    loaded_state_dict = torch.load('test.pth', map_location='cpu')
    tp.load_state_dict(model, state_dict=loaded_state_dict)
    print(model)

    # test
    out = model(torch.randn(1,3,224,224))
    print(out.shape)
    loss = out.sum()
    loss.backward()
    print(model.aha)
          
if __name__=='__main__':
    test_depgraph()