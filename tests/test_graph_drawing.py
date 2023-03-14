import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch_pruning as tp
from torchvision.models import densenet121, resnet18, googlenet, vgg16_bn
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer, vit_b_16
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        skip = x
        x = self.conv2(x)
        x = self.conv3(x)
        x += skip
        return x

model = densenet121() #densenet121() #resnet18() #densenet121() # Net()

unwrapped_parameters = None
round_to = None
if isinstance(
    model, VisionTransformer
):  # Torchvision uses a static hidden_dim for reshape
    round_to = model.encoder.layers[0].num_heads
    unwrapped_parameters = [model.class_token, model.encoder.pos_embedding]

DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 3, 224, 224), unwrapped_parameters=unwrapped_parameters)
tp.utils.draw_dependency_graph(DG, save_as='draw_dep_graph.png', title=None)
tp.utils.draw_groups(DG, save_as='draw_groups.png', title=None)




