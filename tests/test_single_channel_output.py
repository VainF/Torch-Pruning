import torch
from torch import nn
import torch.nn.functional as F
import torch_pruning as tp

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, 2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 1, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x
    
def test_single_channel_output():
    model = Model()
    example_inputs = torch.randn(1, 3, 224, 224)
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

    all_groups = list(DG.get_all_groups())
    print(all_groups[0])
    assert len(all_groups[0])==3

if __name__ == "__main__":
    test_single_channel_output()