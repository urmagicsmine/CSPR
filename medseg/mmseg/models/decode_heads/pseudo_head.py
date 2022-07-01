import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn import normal_init

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class PseudoHead(BaseDecodeHead):
    """ This head is used for Unet3D.
        The Unet3D implementation contains encoder, decoder-head and seg-head.
    To load the pretrained model provided by the official impl conveniently,
    we keep its original structure, and construct this pseudo head, to adapt
    with mm-segmentation code base.
    """

    def __init__(self,
                 **kwargs):
        super(PseudoHead, self).__init__(**kwargs)

    def forward(self, inputs):
        """Forward function."""
        return inputs
