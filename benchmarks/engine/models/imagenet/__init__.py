from torchvision.models import (
    resnet50, 
    densenet121,
    mobilenet_v2,
    googlenet,
    inception_v3,
    squeezenet1_1,
    vgg16_bn,
    vgg19_bn,
    mnasnet1_0,
    alexnet,
)

try:
    from torchvision.models import regnet_x_1_6gf
    from torchvision.models import resnext50_32x4d
    from .vision_transformer import vit_b_16
except:
    regnet_x_1_6gf = None
    resnext50_32x4d = None
    vit_b_16 = None