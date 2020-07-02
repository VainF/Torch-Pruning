from torchvision.models import alexnet
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch_pruning as pruning
import numpy as np 
import torch
import torch.nn.functional as F

model = alexnet(pretrained=True)
print("Before pruning: ")
print(model.features[:4])
print(model.features[0].weight.shape)
print(model.features[3].weight.shape)

pruning.prune_conv(model.features[0], idxs=[0,1,3,4])
pruning.prune_related_conv( model.features[3], idxs=[0,1,3,4] )

print("\nAfter pruning: ")
print(model.features[:4])
print(model.features[0].weight.shape)
print(model.features[3].weight.shape)

mask1 = np.random.randint(low=0, high=2, size=model.features[0].weight.shape)
pruning.mask_weight( model.features[0],mask1 )
print("add mask1, masking %d weights"%( (mask1!=0).sum() ))

mask2 = np.random.randint(low=0, high=2, size=model.features[0].weight.shape)
pruning.mask_weight( model.features[0], mask2)
print("add mask2, masking %d weights"%( (mask2!=0).sum() ))

print("%d weights were actually masked"%( (model.features[0].weight_mask.numpy()!=0).sum() ))
print( "mask1 | mask2 == weight_mask: ", np.alltrue( np.logical_or(mask1, mask2) == model.features[0].weight_mask.numpy() ) )

random_inputs = torch.randn((1,3,224,224))
output = model(random_inputs)

conv1_output = model.features[0](random_inputs) 
masked_weight = torch.tensor( np.logical_or(mask1, mask2) ) *  model.features[0].weight
conv1_output_target = F.conv2d(random_inputs, masked_weight, bias=model.features[0].bias, stride=4, padding=2) 
print( "Correct output from conv1:", torch.all( conv1_output == conv1_output_target ) )