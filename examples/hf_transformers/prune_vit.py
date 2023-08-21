from transformers import ViTImageProcessor, ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTSelfAttention
import torch_pruning as tp
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
example_inputs = processor(images=image, return_tensors="pt")["pixel_values"]
#outputs = model(example_inputs)
#logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
#predicted_class_idx = logits.argmax(-1).item()
#print("Predicted class:", model.config.id2label[predicted_class_idx])

print(model)
imp = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")
base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
channel_groups = {}

# All heads should be pruned simultaneously, so we group channels by head.
for m in model.modules():
    if isinstance(m, ViTSelfAttention):
        channel_groups[m.query] = m.num_attention_heads
        channel_groups[m.key] = m.num_attention_heads
        channel_groups[m.value] = m.num_attention_heads

pruner = tp.pruner.MagnitudePruner(
                model, 
                example_inputs, 
                global_pruning=False, # If False, a uniform sparsity will be assigned to different layers.
                importance=imp, # importance criterion for parameter selection
                iterative_steps=1, # the number of iterations to achieve target sparsity
                ch_sparsity=0.2,
                channel_groups=channel_groups,
                output_transform=lambda out: out.logits.sum(),
                ignored_layers=[model.classifier],
            )

for g in pruner.step(interactive=True):
    #print(g)
    g.prune()

# Modify the attention head size and all head size aftering pruning
for m in model.modules():
    if isinstance(m, ViTSelfAttention):
        m.attention_head_size = m.query.out_features // m.num_attention_heads
        m.all_head_size = m.query.out_features

print(model)
test_output = model(example_inputs)
pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
print("Base MACs: %f G, Pruned MACs: %f G"%(base_macs/1e9, pruned_macs/1e9))
print("Base Params: %f M, Pruned Params: %f M"%(base_params/1e6, pruned_params/1e6))