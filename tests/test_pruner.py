import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import densenet121 as entry
import torch_pruning as tp

model = entry(pretrained=True)
print(model)
# Global metrics
example_inputs = torch.randn(1, 3, 224, 224)
imp = tp.importance.MagnitudeImportance(p=2)
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m)

total_steps = 5
pruner = tp.pruner.LocalMagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    total_steps=total_steps,
    ch_sparsity=0.5,
    ignored_layers=ignored_layers,
)

for i in range(total_steps):
    ori_size = tp.utils.count_params(model)
    pruner.step()
    print(
        "  Params: %.2f M => %.2f M"
        % (ori_size / 1e6, tp.utils.count_params(model) / 1e6)
    )

with torch.no_grad():
    print(model)
    print(model(example_inputs).shape)
