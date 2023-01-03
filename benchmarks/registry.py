from pyexpat import model
from torchvision import datasets, transforms as T
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
import os, sys
import engine.models as models
import engine.utils as utils
from functools import partial
NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'cifar10_224':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100_224': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
}


MODEL_DICT = {
    'resnet18': models.cifar.resnet.resnet18,
    'resnet34': models.cifar.resnet.resnet34,
    'resnet50': models.cifar.resnet.resnet50,
    'resnet101': models.cifar.resnet.resnet101,
    'resnet152': models.cifar.resnet.resnet152,

    'vgg11': models.cifar.vgg.vgg11_bn,
    'vgg13': models.cifar.vgg.vgg13_bn,
    'vgg16': models.cifar.vgg.vgg16_bn,
    'vgg19': models.cifar.vgg.vgg19_bn,

    'densenet121': models.cifar.densenet.densenet121,
    'densenet161': models.cifar.densenet.densenet161,
    'densenet169': models.cifar.densenet.densenet169,
    'densenet201': models.cifar.densenet.densenet201,

    'googlenet': models.cifar.googlenet.googlenet,

    'nasnet': models.cifar.nasnet.nasnet,

    'inceptionv4': models.cifar.inceptionv4.inceptionv4,
    'inceptionv3': models.cifar.inceptionv3.inception_v3,

    'mobilenetv2': models.cifar.mobilenetv2.mobilenetv2,
    
    'preactresnet18': models.cifar.preactresnet.preactresnet18,
    'preactresnet34': models.cifar.preactresnet.preactresnet34,
    'preactresnet50': models.cifar.preactresnet.preactresnet50,
    'preactresnet101': models.cifar.preactresnet.preactresnet101,
    'preactresnet152': models.cifar.preactresnet.preactresnet152,

    #'resnet14': models.cifar.resnet_tiny.resnet14,
    'resnet20': models.cifar.resnet_tiny.resnet20,
    'resnet32': models.cifar.resnet_tiny.resnet32,
    'resnet44': models.cifar.resnet_tiny.resnet44,
    'resnet56': models.cifar.resnet_tiny.resnet56,
    'resnet110': models.cifar.resnet_tiny.resnet110,
    #'resnet8x4': models.cifar.resnet_tiny.resnet8x4,
    #'resnet32x4': models.cifar.resnet_tiny.resnet32x4,

    'resnext50': models.cifar.resnext.resnext50,
    'resnext101': models.cifar.resnext.resnext101,
    'resnext152': models.cifar.resnext.resnext152,
    
    'se_resnet20': models.cifar.senet.se_resnet20,
    'se_resnet32': models.cifar.senet.se_resnet32,
    'se_resnet56': models.cifar.senet.se_resnet56,
    'se_resnet110': models.cifar.senet.se_resnet110,
    'se_resnet164': models.cifar.senet.se_resnet164,

    'xception': models.cifar.xception.xception,
    
    'vit_cifar': models.cifar.vit.vit_cifar,
    'swin_t': models.cifar.swin.swin_t,
    'swin_s': models.cifar.swin.swin_s,
    'swin_b': models.cifar.swin.swin_b,
    'swin_l': models.cifar.swin.swin_l,
}

IMAGENET_MODEL_DICT={
    "resnet50": models.imagenet.resnet50, 
    "densenet121": models.imagenet.densenet121,
    "mobilenet_v2": models.imagenet.mobilenet_v2,
    "mobilenet_v2_w_1_4": partial( models.imagenet.mobilenet_v2,  width_mult=1.4 ),
    "googlenet": models.imagenet.googlenet,
    "inception_v3": models.imagenet.inception_v3,
    "squeezenet1_1": models.imagenet.squeezenet1_1,
    "vgg19_bn": models.imagenet.vgg19_bn,
    "vgg16_bn": models.imagenet.vgg16_bn,
    "mnasnet1_0": models.imagenet.mnasnet1_0,
    "alexnet": models.imagenet.alexnet,
    "regnet_x_1_6gf": models.imagenet.regnet_x_1_6gf,
    "resnext50_32x4d": models.imagenet.resnext50_32x4d,
    "vit_b_16": models.imagenet.vit_b_16,
}

GRAPH_MODEL_DICT = {
    'pointnet': models.graph.pointnet,
    'dgcnn': models.graph.dgcnn,
}

def get_model(name: str, num_classes, pretrained=False, target_dataset='cifar', **kwargs):
    if target_dataset == "imagenet":
        
        model = IMAGENET_MODEL_DICT[name](pretrained=pretrained)
    elif 'cifar' in target_dataset:
        model = MODEL_DICT[name](num_classes=num_classes)
    elif target_dataset == 'modelnet40':
        model = GRAPH_MODEL_DICT[name](num_classes=num_classes)
    return model 


def get_dataset(name: str, data_root: str='data', return_transform=False):
    name = name.lower()
    data_root = os.path.expanduser( data_root )

    if name=='cifar10':
        num_classes = 10
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' )
        train_dst = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=False, transform=val_transform)
        input_size = (1, 3, 32, 32)
    elif name=='cifar100':
        num_classes = 100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)
        input_size = (1, 3, 32, 32)
    elif name=='modelnet40':
        num_classes=40
        train_dst = utils.datasets.ModelNet40(data_root=data_root, partition='train', num_points=1024)
        val_dst = utils.datasets.ModelNet40(data_root=data_root, partition='test', num_points=1024)
        train_transform = val_transform = None
        input_size = (1, 3, 2048)
    else:
        raise NotImplementedError
    if return_transform:
        return num_classes, train_dst, val_dst, input_size, train_transform, val_transform
    return num_classes, train_dst, val_dst, input_size

