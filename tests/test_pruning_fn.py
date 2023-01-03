from torchvision.models import alexnet
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch_pruning as tp

model = alexnet(pretrained=True)
print("Before pruning: ")
print(model.features[:4])
print(model.features[0].weight.shape)
print(model.features[3].weight.shape)

tp.prune_conv_out_channels(model.features[0], idxs=[0, 1, 3, 4])
tp.prune_conv_in_channels(model.features[3], idxs=[0, 1, 3, 4])

print("\nAfter pruning: ")
print(model.features[:4])
print(model.features[0].weight.shape)
print(model.features[3].weight.shape)