import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import densenet121 as entry
import torch_pruning as tp
from torch import nn
import torch.nn.functional as F

def test_pruner():
    model = entry(pretrained=True)
    print(model)
    # Global metrics
    example_inputs = torch.randn(1, 3, 224, 224)

    for imp_cls, pruner_cls in [
        [tp.importance.GroupMagnitudeImportance, tp.pruner.GroupNormPruner],
        [tp.importance.BNScaleImportance, tp.pruner.BNScalePruner],
        [tp.importance.GroupMagnitudeImportance, tp.pruner.GrowingRegPruner],
    ]:
        if imp_cls == tp.importance.OBDCImportance:
            imp = imp_cls(num_classes=1000)
        else:
            imp = imp_cls()
        ignored_layers = []
        # DO NOT prune the final classifier!
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
                ignored_layers.append(m)
        iterative_steps = 5
        pruner = pruner_cls(
            model,
            example_inputs,
            importance=imp,
            global_pruning=True,
            iterative_steps=iterative_steps,
            pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers,
        )
        
        for i in range(iterative_steps):
            if isinstance(imp, tp.importance.OBDCImportance):
                imp._prepare_model(model, pruner)
                model(example_inputs).sum().backward()
                imp.step()
            else:
                model(example_inputs).sum().backward()
            grad_dict = {}
            for p in model.parameters():
                if p.grad is not None:
                    grad_dict[p] = p.grad.clone()
                else:
                    grad_dict[p] = None
            pruner.update_regularizer()
            pruner.regularize(model)
            for name, p in model.named_parameters():
                if p.grad is not None and grad_dict[p] is not None:
                    print(name, (grad_dict[p] - p.grad).abs().sum())
                else:
                    print(name, "has no grad")
            for g in pruner.step(interactive=True):
                g.prune()
            if isinstance(imp, tp.importance.OBDCImportance):
                imp._rm_hooks(model)
                imp._clear_buffer()

    
if __name__ == "__main__":
    test_pruner()
