import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch_pruning as tp
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Linear = nn.Linear(in_features=512, out_features=4096)
        self.conv1T1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv1T2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.final = nn.Linear(16384, 10)
    def forward(self, x):
        x = F.relu(self.Linear(x))
        x = x.view(-1,self.conv1T1.in_channels, 4, 4)
        x = F.relu(self.conv1T1(x))
        x = F.relu(self.conv1T2(x))
        x = torch.flatten(x, 1)
        x = self.final(x)
        return x

def test_reshape():
    model = Net()
    example_inputs = torch.randn(1, 512)
    imp = tp.importance.MagnitudeImportance()
    ignored_layers = [model.final]

    iterative_steps = 5
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
        root_module_types=[nn.ConvTranspose2d, nn.Linear],
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(model)
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

if __name__=='__main__':
    test_reshape()

