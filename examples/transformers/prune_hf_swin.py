from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
import torch
import torch.nn as nn
from typing import Sequence
import torch_pruning as tp
from transformers.models.swin.modeling_swin import SwinSelfAttention, SwinPatchMerging


class SwinPatchMergingPruner(tp.BasePruningFunc):

    def prune_out_channels(self, layer: nn.Module, idxs: list):
        tp.prune_linear_out_channels(layer.reduction, idxs)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        dim = layer.dim
        idxs_repeated = idxs + \
            [i+dim for i in idxs] + \
            [i+2*dim for i in idxs] + \
            [i+3*dim for i in idxs]
        tp.prune_linear_in_channels(layer.reduction, idxs_repeated)
        tp.prune_layernorm_out_channels(layer.norm, idxs_repeated)
        return layer

    def get_out_channels(self, layer):
        return layer.reduction.out_features

    def get_in_channels(self, layer):
        return layer.dim

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

example_inputs = processor(images=image, return_tensors="pt")["pixel_values"]
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])


print(model)
imp = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")
base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
num_heads = {}

ignored_layers = [model.classifier]
# All heads should be pruned simultaneously, so we group channels by head.
for m in model.modules():
    if isinstance(m, SwinSelfAttention):
        num_heads[m.query] = m.num_attention_heads
        num_heads[m.key] = m.num_attention_heads
        num_heads[m.value] = m.num_attention_heads

pruner = tp.pruner.BasePruner(
                model, 
                example_inputs, 
                global_pruning=False, # If False, a uniform pruning ratio will be assigned to different layers.
                importance=imp, # importance criterion for parameter selection
                iterative_steps=1, # the number of iterations to achieve target pruning ratio
                pruning_ratio=0.5,
                num_heads=num_heads,
                output_transform=lambda out: out.logits.sum(),
                ignored_layers=ignored_layers,
                customized_pruners={SwinPatchMerging: SwinPatchMergingPruner()},
                root_module_types=(nn.Linear, nn.LayerNorm, SwinPatchMerging),
            )

for g in pruner.step(interactive=True):
    #print(g)
    g.prune()

print(model)

# Modify the attention head size and all head size aftering pruning
for m in model.modules():
    if isinstance(m, SwinSelfAttention):
        m.attention_head_size = m.query.out_features // m.num_attention_heads
        m.all_head_size = m.query.out_features

test_output = model(example_inputs)
pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
print("Base MACs: %f G, Pruned MACs: %f G"%(base_macs/1e9, pruned_macs/1e9))
print("Base Params: %f M, Pruned Params: %f M"%(base_params/1e6, pruned_params/1e6))


