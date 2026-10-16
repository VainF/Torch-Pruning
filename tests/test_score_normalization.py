import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import pytest
from torchvision.models import resnet18 as entry
import torch_pruning as tp
from torch import nn
import torch.nn.functional as F

def test_pruner():
    
    # Global metrics
    example_inputs = torch.randn(1, 3, 224, 224)
    for normalization in ['mean', 'max', 'gaussian', 'standarization', 'sum', 'standardization']:
        model = entry()
        print(model)
        imp = tp.importance.MagnitudeImportance(p=2, normalizer=normalization)
        ignored_layers = []

        # DO NOT prune the final classifier!
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
                ignored_layers.append(m)

        iterative_steps = 5
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            global_pruning=True,
            iterative_steps=iterative_steps,
            pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers,
        )

        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        for i in range(iterative_steps):
            pruner.step()
            macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
            print(model)
            print(model(example_inputs).shape)
            print(
                "  Iter %d/%d, Params: %.2f M => %.2f M"
                % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
            )
            print(
                "  Iter %d/%d, MACs: %.2f G => %.2f G"
                % (i+1, iterative_steps, base_macs / 1e9, macs / 1e9)
            )
            # finetune your model here
            # finetune(model)
            # ...

@pytest.mark.parametrize("normalizer", ["standardization", "standarization"])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("weights", [[-1., 2., 4.], [2., 2., 2.], [0., 0., 0.]])
def test_standardization(normalizer, p, weights):
    model = nn.Linear(1, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(weights).reshape(3, 1))
    graph = tp.DependencyGraph().build_dependency(model, torch.ones(1, 1))
    group = graph.get_pruning_group(model, tp.prune_linear_out_channels, idxs=[0, 1, 2])

    scores = tp.importance.MagnitudeImportance(p=p, normalizer=normalizer)(group)
    raw_scores = torch.tensor(weights).abs().pow(p)
    expected = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
    torch.testing.assert_close(scores, expected)


if __name__ == "__main__":
    test_pruner()
