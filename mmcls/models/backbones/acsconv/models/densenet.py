"""
DenseNet121
Difference from densenet in torchvision for higher resolution:
1. Modify the stride of first convolution layer (7x7 with stride 2) into 1  
2. Remove the first max-pooling layer
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

from ..operators import ACSConv


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        model_state_dict = model.state_dict()
        online_sd = list(state_dict.items())
        count = 0
        for i, k in enumerate(model_state_dict.keys()):
            if 'num_batches_tracked' not in k:
                print(i, count, k, online_sd[count][0])
                model_state_dict[k] = online_sd[count][1]
                count += 1
        model.load_state_dict(model_state_dict)
        print('densenet loaded imagenet pretrained weights')
    else:
        print('densenet without imagenet pretrained weights')
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', ACSConv(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', ACSConv(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, downsample=True):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', ACSConv(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        if downsample:
            self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', ACSConv(3, num_init_features, kernel_size=7, stride=1,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            setattr(self, 'layer{}'.format(i+1), nn.Sequential(OrderedDict([
                ('denseblock%d' % (i + 1), _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate)),
            ])))
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                if i in [0,1,2]:
                    downsample = True
                else:
                    downsample = False
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2, downsample=downsample)
                getattr(self, 'layer{}'.format(i+1)).add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        for m in self.modules():
            if isinstance(m, ACSConv):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
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

class FCNDenseNet(nn.Module):
    def __init__(self, pretrained, num_classes, backbone='densenet121'):
        super().__init__()
        self.backbone = globals()[backbone](pretrained=pretrained)
        self.conv1 = ACSConv(1024+256, 256, kernel_size=1, stride=1,
                            padding=0, bias=False)
        self.conv2 = ACSConv(256+64, 64, kernel_size=1, stride=1,
                            padding=0, bias=False)
        self.classifier = FCNHead(in_channels=64, channels=num_classes)
    
    def forward(self, x):
        features, features1, features2 = self.backbone(x)
        # print(features.shape, features1.shape, features2.shape)
        features_cat1 = torch.cat([features1, F.interpolate(features, scale_factor=2, mode='trilinear')], dim=1)
        features_cat1 = self.conv1(features_cat1)
        features_cat2 = torch.cat([features2, F.interpolate(features_cat1, scale_factor=4, mode='trilinear')], dim=1)
        features_cat2 = self.conv2(features_cat2)
        features = features_cat2

        out = self.classifier(features)
        return out

class ClsDenseNet(nn.Module):
    def __init__(self, pretrained, num_classes, backbone='densenet121'):
        super().__init__()
        self.backbone = globals()[backbone](pretrained=pretrained)
        self.fc = nn.Linear(1024, num_classes, bias=True)
    
    def forward(self, x):
        features = self.backbone(x)[0]
        features = F.adaptive_avg_pool3d(features, output_size=1).view(features.shape[0], -1)
        out = self.fc(features)
        return out
