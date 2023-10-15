import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet50 as entry
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

    iterative_steps = 1
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        global_pruning=True,
        iterative_steps=iterative_steps,
        pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    base_macs, base_nparams, base_layer_macs, base_layer_params = tp.utils.count_ops_and_params(model, example_inputs, layer_wise=True)
    for i in range(iterative_steps):
        pruner.step()
        macs, nparams, layer_macs, layer_params = tp.utils.count_ops_and_params(model, example_inputs, layer_wise=True)
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
        
        for name, module in model.named_modules():
            if name=='': 
                name = 'ALL'
            print(name, layer_macs[module]/1e9, "G ", layer_params[module]/1e6, "M")

        # finetune your model here
        # finetune(model)
        # ...

if __name__=='__main__':
    test_pruner()