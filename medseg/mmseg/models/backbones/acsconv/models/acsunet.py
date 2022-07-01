import torch
import torch.nn.functional as F
from torch import nn
from ..operators import ACSConv

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            ACSConv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            ACSConv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            ACSConv(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            ACSConv(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.decode(x)


class ACSUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.enc1 = _EncoderBlock(1, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.dec1 = _DecoderBlock(192, 64, 32)
        self.interpolate = nn.Upsample(scale_factor=2, mode='trilinear')
        self.final = ACSConv(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.enc1(x)
        x1 = x.clone()
        x = self.enc2(x)
        x = self.dec1(torch.cat([x1, self.interpolate(x)], 1))
        x = self.final(self.interpolate(x))
        return x