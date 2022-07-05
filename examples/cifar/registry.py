from pyexpat import model
from torchvision import datasets, transforms as T
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
import os
import models

NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'cifar10_224':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100_224': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
}


MODEL_DICT = {
    'resnet18': models.resnet.resnet18,
    'resnet34': models.resnet.resnet34,
    'resnet50': models.resnet.resnet50,
    'resnet101': models.resnet.resnet101,
    'resnet152': models.resnet.resnet152,

    'vgg11': models.vgg.vgg11_bn,
    'vgg13': models.vgg.vgg13_bn,
    'vgg16': models.vgg.vgg16_bn,
    'vgg19': models.vgg.vgg19_bn,

    'densenet121': models.densenet.densenet121,
    'densenet161': models.densenet.densenet161,
    'densenet169': models.densenet.densenet169,
    'densenet201': models.densenet.densenet201,

    'googlenet': models.googlenet.googlenet,

    'inceptionv4': models.inceptionv4.inceptionv4,
    
    'mobilenetv2': models.mobilenetv2.mobilenetv2,
    
    'preactresnet18': models.preactresnet.preactresnet18,
    'preactresnet34': models.preactresnet.preactresnet34,
    'preactresnet50': models.preactresnet.preactresnet50,
    'preactresnet101': models.preactresnet.preactresnet101,
    'preactresnet152': models.preactresnet.preactresnet152,

    'resnet14': models.resnet_tiny.resnet14,
    'resnet20': models.resnet_tiny.resnet20,
    'resnet32': models.resnet_tiny.resnet32,
    'resnet44': models.resnet_tiny.resnet44,
    'resnet56': models.resnet_tiny.resnet56,
    'resnet110': models.resnet_tiny.resnet110,
    'resnet8x4': models.resnet_tiny.resnet8x4,
    'resnet32x4': models.resnet_tiny.resnet32x4,

    'resnext50': models.resnext.resnext50,
    'resnext101': models.resnext.resnext101,
    'resnext152': models.resnext.resnext152,

    'vit_cifar': models.vit.vit_cifar,
    'swin_t': models.swin.swin_t,
    'swin_s': models.swin.swin_s,
    'swin_b': models.swin.swin_b,
    'swin_l': models.swin.swin_l,

}


def get_model(name: str, num_classes, pretrained=False, **kwargs):
    model = MODEL_DICT[name](num_classes=num_classes)
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
        train_dst = datasets.CIFAR10(data_root, train=True, download=False, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=False, transform=val_transform)
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
    elif name=='cifar10_224':
        num_classes = 10
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.Resize(224),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' )
        train_dst = datasets.CIFAR10(data_root, train=True, download=False, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=False, transform=val_transform)
    elif name=='cifar100_224':
        num_classes = 100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.Resize(224),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)
    else:
        raise NotImplementedError
    if return_transform:
        return num_classes, train_dst, val_dst, train_transform, val_transform
    return num_classes, train_dst, val_dst