import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet18
import torch_pruning as tp

def test_depgraph():
    model = resnet18().eval()

    # 1. build dependency graph for resnet18
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))

    # 2. Select channels for pruning, here we prune the channels indexed by [2, 6, 9].
    pruning_idxs = pruning_idxs=[2, 6, 9]
    pruning_group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs )

    print("Pruning Group:")
    print(pruning_group.details()) # or print(pruning_group)

    # 3. prune all grouped layer that is coupled with model.conv1
    if DG.check_pruning_group(pruning_group):
        pruning_group.prune()

    print("After pruning:")
    print(model)

    for group in DG.get_all_groups():
        # handle groups in sequential order
        idxs = [2,4,6] # my pruning indices
        group.prune(idxs=idxs)
    print(model)

    #groups = list(DG.get_all_groups())
    #print("Num groups: %d"%(len(groups)))

    #for g in groups:
    #    print(g)

def test_tensor_input_preserves_batch_dimension():
    inputs = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    for input_kind in ("tensor", "tuple", "list", "dict", "forward_fn"):
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.ReLU())
        with torch.no_grad():
            model[0].weight.copy_(torch.arange(12).reshape(4, 3, 1, 1) / 12)
            model[0].bias.copy_(torch.arange(4) / 10)
        example_inputs = {
            "tensor": inputs,
            "tuple": (inputs,),
            "list": [inputs],
            "dict": {"input": inputs},
            "forward_fn": {"tensor": inputs},
        }[input_kind]
        forward_fn = None
        if input_kind == "forward_fn":
            forward_fn = lambda model, x: model(x["tensor"] + 1)
        expected = model(inputs + 1 if forward_fn else inputs).detach()[:, 2:]
        graph = tp.DependencyGraph().build_dependency(
            model, example_inputs=example_inputs, forward_fn=forward_fn
        )
        group = graph.get_pruning_group(model[0], tp.prune_conv_out_channels, idxs=[0, 1])
        assert graph.check_pruning_group(group), input_kind
        group.prune()
        actual = model(inputs + 1 if forward_fn else inputs)
        assert actual.shape == (1, 2, 2, 2)
        assert torch.allclose(actual, expected)


if __name__ == "__main__":
    test_depgraph()
