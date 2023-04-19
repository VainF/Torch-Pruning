import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet18 as entry
import torch_pruning as tp
from torch import nn
import torch.nn.functional as F

def test_pruner():
    model = entry()
    print(model)
    # Global metrics
    example_inputs = torch.randn(1, 3, 224, 224)
    imp = tp.importance.MagnitudeImportance(p=2)
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
        iterative_steps=iterative_steps,
        ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
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

def test_pruner_TransposeConv():
    class TransposeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.ConvTranspose2d(16, 64, 2, stride=2)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 8, 3, 1, 1)
            self.bn3 = nn.BatchNorm2d(8)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            return x

    # Global metrics
    example_inputs = torch.randn(1, 3, 224, 224)

    ignored_layers = []

    # DO NOT prune the final classifier!
    ignored_layers = []

    iterative_steps = 5
    for imp in [
        tp.importance.MagnitudeImportance(p=2),
        tp.importance.LAMPImportance(p=2),
        tp.importance.BNScaleImportance(),
        tp.importance.GroupNormImportance(p=2),
    ]:
        for Pruner in [
            tp.pruner.MagnitudePruner,
            tp.pruner.GroupNormPruner,
            tp.pruner.BNScalePruner,
        ]:
            model = TransposeModel()
            print("imp: ", type(imp))
            print("before: ", model)

            print(Pruner.__name__)
            pruner = Pruner(
                model,
                example_inputs,
                importance=imp,
                iterative_steps=iterative_steps,
                ch_sparsity=0.5,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
                ignored_layers=ignored_layers,
            )

            base_macs, base_nparams = tp.utils.count_ops_and_params(
                model, example_inputs
            )
            for i in range(iterative_steps):
                pruner.step()
                macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
                print(model(example_inputs).shape)
                print(
                    "  Iter %d/%d, Params: %.2f M => %.2f M"
                    % (i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
                )
                print(
                    "  Iter %d/%d, MACs: %.2f G => %.2f G"
                    % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9)
                )
            print("after: ", model)


if __name__ == "__main__":
    test_pruner()
    test_pruner_TransposeConv()
