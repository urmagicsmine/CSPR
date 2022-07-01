"""
VGG16 with BN
Difference from vgg16 in torchvision for higher resolution:
1. Layer config changes from [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'] (torchvision)
                          to [64, 64, 128, 128, 'M', 256, 256, 256, 512, 512, 512, 'M', 512, 512, 512] (acsconv)
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from collections import OrderedDict
from torch.nn import functional as F

from ..operators import ACSConv

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG16_bn(nn.Module):

    def __init__(self, pretrained=False, num_classes=1000, init_weights=True):
        super(VGG16_bn, self).__init__()

        self.layer0 = nn.Sequential(OrderedDict([
            ('conv0', ACSConv(3, 64, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm3d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', ACSConv(64, 64, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm3d(64)),
            ('relu1', nn.ReLU(inplace=True))
        ]))
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv0', ACSConv(64, 128, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm3d(128)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', ACSConv(128, 128, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm3d(128)),
            ('relu1', nn.ReLU(inplace=True))
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('maxpool0', nn.MaxPool3d(kernel_size=2, stride=2)),
            ('conv0', ACSConv(128, 256, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm3d(256)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', ACSConv(256, 256, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm3d(256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', ACSConv(256, 256, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm3d(256)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.layer3 = nn.Sequential(OrderedDict([
            ('conv0', ACSConv(256, 512, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm3d(512)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', ACSConv(512, 512, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm3d(512)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', ACSConv(512, 512, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm3d(512)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.layer4 = nn.Sequential(OrderedDict([
            ('maxpool0', nn.MaxPool3d(kernel_size=2, stride=2)),
            ('conv0', ACSConv(512, 512, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm3d(512)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', ACSConv(512, 512, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm3d(512)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', ACSConv(512, 512, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm3d(512)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))

        self._initialize_weights()
        if pretrained:
            model_state_dict = self.state_dict()
            online_sd = list(load_state_dict_from_url(model_urls['vgg16_bn']).items())
            count = 0
            for i, k in enumerate(model_state_dict.keys()):
                if 'num_batches_tracked' not in k:
                    print(i, count, k, online_sd[count][0])
                    model_state_dict[k] = online_sd[count][1]
                    count += 1
            self.load_state_dict(model_state_dict)
            print('vgg loaded imagenet pretrained weights')
        else:
            print('vgg without imagenet pretrained weights')
    def forward(self, x):
        x = self.layer0(x)
        x2 = x.clone()
        x = self.layer1(x)
        x = self.layer2(x)
        x1 = x.clone()
        x = self.layer3(x)
        x = self.layer4(x)

        return x, x1, x2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ACSConv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg16(*args, **kwargs):
    return VGG16_bn(*args, **kwargs)

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

class FCNVGG(nn.Module):
    def __init__(self, pretrained, num_classes, backbone='VGG16_bn'):
        super().__init__()
        self.backbone = globals()[backbone](pretrained=pretrained)
        self.conv1 = ACSConv(512+256, 256, kernel_size=1, stride=1,
                            padding=0, bias=False)
        self.conv2 = ACSConv(256+64, 64, kernel_size=1, stride=1,
                            padding=0, bias=False)
        self.classifier = FCNHead(in_channels=64, channels=num_classes)
    
    def forward(self, x):
        features, features1, features2 = self.backbone(x)
        # print(features.shape, features1.shape, features2.shape)
        features_cat1 = torch.cat([features1, F.interpolate(features, scale_factor=2, mode='trilinear')], dim=1)
        features_cat1 = self.conv1(features_cat1)
        features_cat2 = torch.cat([features2, F.interpolate(features_cat1, scale_factor=2, mode='trilinear')], dim=1)
        features_cat2 = self.conv2(features_cat2)
        features = features_cat2

        out = self.classifier(features)
        return out


class ClsVGG(nn.Module):
    def __init__(self, pretrained, num_classes, backbone='vgg16'):
        super().__init__()
        self.backbone = globals()[backbone](pretrained=pretrained)
        self.fc = nn.Linear(512, num_classes, bias=True)
    
    def forward(self, x):
        features = self.backbone(x)[0]
        features = F.adaptive_avg_pool3d(features, output_size=1).view(features.shape[0], -1)
        out = self.fc(features)
        return out
