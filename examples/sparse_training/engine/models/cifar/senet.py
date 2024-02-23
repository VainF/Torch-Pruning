"""
The script is adapted from torchvision.models.ResNet
"""

import torch.nn as nn

__all__ = ['se_resnet20', 'se_resnet32', 'se_resnet44', 'se_resnet56', 'se_resnet110', 'se_resnet164',
           'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet164']

model_urls = {
    'se_resnet18': None,
    'se_resnet34': None,
    'se_resnet50': None,
    'se_resnet101': None,
    'se_resnet152': None,
}

BN_momentum = 0.1


class SqueezeExcitationLayer(nn.Module):
    def __init__(self,
                 channel,
                 reduction=16):
        super(SqueezeExcitationLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, new_resnet=False):
        super(BasicBlock, self).__init__()
        self.new_resnet = new_resnet
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(
            inplanes if new_resnet else planes, momentum=BN_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_momentum)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes, momentum=BN_momentum))
        else:
            self.downsample = lambda x: x
        self.stride = stride
        self.output = planes * self.expansion

    def _old_resnet(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def _new_resnet(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out

    def forward(self, x):
        if self.new_resnet:
            return self._new_resnet(x)
        else:
            return self._old_resnet(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, reduction=16, new_resnet=False):
        super(SEBasicBlock, self).__init__()
        self.new_resnet = new_resnet
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(
            inplanes if new_resnet else planes, momentum=BN_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_momentum)
        self.se = SqueezeExcitationLayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes, momentum=BN_momentum))
        else:
            self.downsample = lambda x: x
        self.stride = stride
        self.output = planes * self.expansion

    def _old_resnet(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def _new_resnet(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

    def forward(self, x):
        if self.new_resnet:
            return self._new_resnet(x)
        else:
            return self._old_resnet(x)


class CifarNet(nn.Module):
    """
    This is specially designed for cifar10
    """

    def __init__(self, block, n_size, num_classes=10, reduction=16, new_resnet=False, dropout=0.):
        super(CifarNet, self).__init__()
        self.inplane = 16
        self.new_resnet = new_resnet
        self.dropout_prob = dropout
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane, momentum=BN_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(
            block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(
            block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if self.dropout_prob > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout_prob, inplace=True)
        self.fc = nn.Linear(self.inplane, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride,
                                reduction, new_resnet=self.new_resnet))
            self.inplane = layers[-1].output

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout_prob > 0:
            x = self.dropout_layer(x)
        x = self.fc(x)

        return x


def se_resnet20(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = CifarNet(SEBasicBlock, 3, **kwargs)
    return model


def se_resnet32(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarNet(SEBasicBlock, 5, **kwargs)
    return model


def se_resnet44(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarNet(SEBasicBlock, 7, **kwargs)
    return model


def se_resnet56(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarNet(SEBasicBlock, 9, **kwargs)
    return model


def se_resnet110(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarNet(SEBasicBlock, 18, **kwargs)
    return model


def se_resnet164(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarNet(SEBasicBlock, 27, **kwargs)
    return model


def resnet20(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = CifarNet(BasicBlock, 3, **kwargs)
    return model


def resnet32(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarNet(BasicBlock, 5, **kwargs)
    return model


def resnet44(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarNet(BasicBlock, 7, **kwargs)
    return model


def resnet56(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarNet(BasicBlock, 9, **kwargs)
    return model


def resnet110(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarNet(BasicBlock, 18, **kwargs)
    return model


def resnet164(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarNet(BasicBlock, 27, **kwargs)
    return model