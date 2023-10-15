import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch_pruning as tp
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim)
        )
        self.parallel_path = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim)
        )

        self.conv1 = nn.Conv2d(in_dim, in_dim, 1)
        self.conv2 = nn.Conv2d(in_dim, in_dim, 1)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.parallel_path(x)
        x = torch.cat([x1, x2], dim=2)
        x = self.conv1(x)
        x1, x2 = torch.split(x, [x1.shape[2], x2.shape[2]], dim=2)
        x = self.conv2(x1)
        return x
    
def test_pruner():
    model = Net(512)
    print(model)
    # Global metrics
    example_inputs = torch.randn(1, 512, 7, 7)
    imp = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = []

    # DO NOT prune the final classifier!
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)

    iterative_steps = 1
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):
        pruner.step()
        print(model)
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        
        print(model(example_inputs).shape)
        print(
            "  Iter %d/%d, Params: %.2f => %.2f"
            % (i+1, iterative_steps, base_nparams, nparams)
        )
        print(
            "  Iter %d/%d, MACs: %.2f => %.2f"
            % (i+1, iterative_steps, base_macs, macs)
        )
        # finetune your model here
        # finetune(model)
        # ...

if __name__=='__main__':
    test_pruner()