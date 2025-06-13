import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch_pruning as tp
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0):
        super().__init__()
        if dims is None:
            dims = []
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class widedeep(nn.Module):
    def __init__(self, input_dim):
        super(widedeep, self).__init__()
        self.dims = input_dim

        self.mlp = MLP(self.dims, True, dims=[32,16], dropout=0.2)
        self.linear = nn.Linear(self.dims, 3)
        self.lin2 = nn.Linear(4, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        mlp_out = self.mlp(x)
        linear_out = self.linear(x)
        x = torch.concat([linear_out, mlp_out], dim=-1)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x
    
def test_pruner():
    model = widedeep(32)
    print(model)
    # Global metrics
    example_inputs = torch.randn(1, 32)
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