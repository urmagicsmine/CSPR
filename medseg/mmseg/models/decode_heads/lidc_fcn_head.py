import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn import normal_init

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class LIDCFCNHead(BaseDecodeHead):
    """ This head is used for LIDC-IDRI segmentation task.
        With input as (Stem of Resnet, Stage2 of Resnet, Stage4 of Resnet),
    this FCN-like HEAD performs u-net like short-cut OP, upsampling high-level features
    (which have low resolution) and concating them with lower level features(higher resolution),
    and output the seg map with the same shape as input tensors.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 **kwargs):
        super(LIDCFCNHead, self).__init__(**kwargs)

        self.conv1 = ConvModule(self.in_channels[1] + self.in_channels[2], self.channels,
                kernel_size=1, stride=1, padding=0, bias=False,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        self.conv2 = ConvModule(self.in_channels[0] + self.in_channels[2], self.channels,
                kernel_size=1, stride=1, padding=0, bias=False,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        self.conv3 = ConvModule(
            self.channels, self.channels,
            kernel_size=3, padding=1, stride=1,
            conv_cfg=self.conv_cfg,
            #norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        #self.seg_head = FCNHead(self.channels, self.num_classes)
        self.seg_head = nn.Sequential(
            nn.Conv3d(self.channels, self.channels//4, 3, padding=1, bias=False),
            nn.SyncBatchNorm(self.channels//4),
            #nn.BatchNorm3d(self.channels//4),
            #nn.GroupNorm(8, self.channels//4),
            nn.ReLU(),
            nn.Conv3d(self.channels//4, self.num_classes, 1)
        )
        #self.conv_out = nn.Conv3d(self.channels, self.num_classes,
                #kernel_size=1, stride=1, padding=0, bias=False)
        self.init_weights()

    def init_weights(self):
        """Initialize weights of classification layer."""
        #normal_init(self.conv1, mean=0, std=0.01)
        #normal_init(self.conv2, mean=0, std=0.01)
        #normal_init(self.conv3, mean=0, std=0.01)
        #normal_init(self.conv_out, mean=0, std=0.01)
        return

    def forward(self, inputs):
        """Forward function."""
        stem, stage2, stage4= self._transform_inputs(inputs)
        features_cat1 = torch.cat([stage2, F.interpolate(stage4, scale_factor=2)], dim=1)
        features_cat1 = self.conv1(features_cat1)
        features_cat2 = torch.cat([stem, F.interpolate(features_cat1, scale_factor=2)], dim=1)
        output = self.conv2(features_cat2)
        # In ACS, the final pred conv is implemented as conv+BN+relu.
        #output = self.cls_seg(output)
        output = self.seg_head(output)
        #output = self.conv3(output)
        #output = self.conv_out(output)
        return output

class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels, use_GN=False):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(inter_channels) if not use_GN else nn.GroupNorm(32,planes),
            nn.ReLU(),
            nn.Conv3d(inter_channels, channels, 1)
        ]
        super(FCNHead, self).__init__(*layers)
