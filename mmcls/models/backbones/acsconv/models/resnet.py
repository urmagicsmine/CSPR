"""
ResNet18
Difference from ResNet18 in torchvision for higher resolution:
1. Modify the stride of first convolution layer (7x7 with stride 2) into 1  
2. Remove the first max-pooling layer
3. Downsample and upsample twice in encoder and decoder respectively in FCNResNet
"""

import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from ..operators import ACSConv

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet18'])
        for key in list(state_dict.keys()):
            if 'fc' in key:
                del state_dict[key]
        model.load_state_dict(state_dict,strict=False)
        print('resnet18 loaded imagenet pretrained weights')
    else:
        print('resnet18 without imagenet pretrained weights')
    return model

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return ACSConv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = ACSConv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = ACSConv(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = ACSConv(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = ACSConv(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, ACSConv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ACSConv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x2 = x.clone()
        x = self.layer1(x)
        x = self.layer2(x)

        x1 = x.clone()
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x, x1, x2

class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            ACSConv(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(),
            ACSConv(inter_channels, channels, 1)
        ]
        super(FCNHead, self).__init__(*layers)

class FCNResNet(nn.Module):
    def __init__(self, pretrained, num_classes, backbone='resnet18'):
        super().__init__()
        self.backbone = globals()[backbone](pretrained=pretrained)
        self.conv1 = ACSConv((128+512), 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = ACSConv(64+512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.classifier = FCNHead(in_channels=512, channels=num_classes)
    
    def forward(self, x):
        features, features1, features2 = self.backbone(x)
        features_cat1 = torch.cat([features1, F.interpolate(features, scale_factor=2, mode='trilinear')], dim=1)
        features_cat1 = self.conv1(features_cat1)
        features_cat2 = torch.cat([features2, F.interpolate(features_cat1, scale_factor=2, mode='trilinear')], dim=1)
        features_cat2 = self.conv2(features_cat2)
        features = features_cat2

        out = self.classifier(features)
        return out

class ClsResNet(nn.Module):
    def __init__(self, pretrained, num_classes, backbone='resnet18'):
        super().__init__()
        self.backbone = globals()[backbone](pretrained=pretrained)
        self.fc = nn.Linear(512, num_classes, bias=True)
    
    def forward(self, x):
        features = self.backbone(x)[0]
        features = F.adaptive_avg_pool3d(features, output_size=1).view(features.shape[0], -1)
        out = self.fc(features)
        return out
