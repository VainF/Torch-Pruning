import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torchvision.models import resnet18 as entry
import torch_pruning as tp

model = entry(pretrained=False)
model = tp.helpers.gconv2convs(model)
print(model)
# Global metrics
ori_size = tp.utils.count_params(model)
example_inputs = torch.randn(1, 3, 224, 224)
imp = tp.importance.MagnitudeImportance(p=2)
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m)

pruner = tp.pruner.MagnitudeBasedPruner(
    model,
    example_inputs,
    importance=imp,
    steps=5,
    ch_sparsity=0.5,
    ignored_layers=ignored_layers,
)

for i in range(5):
    pruner.step()
    print(
        "  Params: %.2f M => %.2f M"
        % (ori_size / 1e6, tp.utils.count_params(model) / 1e6)
    )
with torch.no_grad():
    print(model)
    print(model(example_inputs).shape)
