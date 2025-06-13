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
        self.block2 = nn.Sequential(
            nn.Conv2d(in_dim, 2*in_dim, 1),
            nn.BatchNorm2d(2*in_dim),
            nn.GELU(),
            nn.Conv2d(2*in_dim, 2*in_dim, 1),
            nn.BatchNorm2d(2*in_dim)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_dim, 3*in_dim, 1),
            nn.BatchNorm2d(3*in_dim)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(in_dim, 6*in_dim, 1),
            nn.BatchNorm2d(6*in_dim)
        )

        self.output = nn.Conv2d(12*in_dim, 10, 1)

    def forward(self, x):
        output1 = self.block1(x)
        output2 = self.block2(x)

        output12 = torch.cat([output1, output2], dim=1)

        output3 = self.block3(x)
        output123 = torch.cat([output12, output3], dim=1)

        output4 = self.block4(x)
        output123_4 = torch.cat([output123, output4], dim=1)
        return self.output(output123_4)
    
def test_pruner():
    model = Net(10)
    
    # Global metrics
    example_inputs = torch.randn(1, 10, 7, 7)
    imp = tp.importance.RandomImportance()
    ignored_layers = []

    iterative_steps = 5
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
        tp.utils.print_tool.before_pruning(model)
        for g in pruner.step(interactive=True):
            #print(g.details())
            g.prune()
        tp.utils.print_tool.after_pruning(model)
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(model(example_inputs).shape)
        print(
            "  Iter %d/%d, Params: %.2f => %.2f"
            % (i+1, iterative_steps, base_nparams, nparams )
        )
        print(
            "  Iter %d/%d, MACs: %.2f => %.2f "
            % (i+1, iterative_steps, base_macs, macs)
        )
    assert model.block1[0].out_channels == 5
    assert model.block2[0].out_channels == 10
    assert model.block3[0].out_channels == 15
    assert model.block4[0].out_channels == 30
        # finetune your model here
        # finetune(model)
        # ...

if __name__=='__main__':
    test_pruner()