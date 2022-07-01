import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from collections import OrderedDict

from ..utils import _to_triple, _triple_same, _pair_same
from .base_acsconv import _ACSConv


class SoftACSConv(_ACSConv):
    """
    Decorator class for soft ACS Convolution

    Args:
        mean: *bool*, optional, the default value is False. If True, it changes to a mean ACS Convolution.

        Other arguments are the same as torch.nn.Conv3d.
    Examples:
        >>> import SoftACSConv
        >>> x = torch.rand(batch_size, 3, D, H, W)
        >>> # soft ACS Convolution
        >>> conv = SoftACSConv(3, 10, 1)
        >>> out = conv(x)

        >>> # mean ACS Convolution
        >>> conv = SoftACSConv(3, 10, 1, mean=Ture)
        >>> out = conv(x)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, mean=False,
                 bias=True, padding_mode='zeros'):

        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, 0, groups, bias, padding_mode)
        if not mean:
            self.soft_w_core = nn.Parameter(torch.rand(out_channels,3)) #TODO: init
        self.mean = mean

    def conv3D_output_shape_f(self,i, input_shape):
        """
        Calculate the original output size assuming the convolution is nn.Conv3d based on 
        input size, kernel size, dilation, padding and stride.
        """
        return math.floor((input_shape[i]-self.kernel_size[i]-(self.dilation[i]-1)*
                                         (self.kernel_size[i]-1)+2*self.padding[i])
                                        /self.stride[i])+1
    
    def forward(self, x):
        """
        Convolution forward function
        Conduct convolution on three directions seperately and then 
        aggregate the three parts of feature maps by *soft* or *mean* way. 
        Bias is added at last.
        """
        B, C_in, *input_shape = x.shape
        conv3D_output_shape = (self.conv3D_output_shape_f(0, input_shape), 
                               self.conv3D_output_shape_f(1, input_shape), 
                               self.conv3D_output_shape_f(2, input_shape))

        
        f_a = F.conv3d(x if conv3D_output_shape[0]==input_shape[0] or 2*conv3D_output_shape[0]==input_shape[0] else F.pad(x, (0,0,0,0,self.padding[0],self.padding[0]),'constant',0)[:,:,
                                                self.kernel_size[0]//2:self.kernel_size[0]//2+(conv3D_output_shape[0]-1)*self.stride[0]+1,
                                                :,:], 
                                                weight=self.weight.unsqueeze(2), bias=None, 
                                                stride=self.stride,
                                                padding=(0,self.padding[1],self.padding[2]),
                                                dilation=self.dilation,
                                                groups=self.groups)
                        
        f_c = F.conv3d(x if conv3D_output_shape[1]==input_shape[1] or 2*conv3D_output_shape[1]==input_shape[1] else F.pad(x, (0,0,self.padding[1],self.padding[1]),'constant',0)[:,:,:,
                                                self.kernel_size[1]//2:self.kernel_size[1]//2+self.stride[1]*(conv3D_output_shape[1]-1)+1,
                                                :],
                                                weight=self.weight.unsqueeze(3), bias=None,                                     
                                                stride=self.stride,
                                                padding=(self.padding[0],0,self.padding[2]),
                                                dilation=self.dilation,
                                                groups=self.groups)

        f_s = F.conv3d(x if conv3D_output_shape[2]==input_shape[2] or 2*conv3D_output_shape[2]==input_shape[2] else F.pad(x, (self.padding[2],self.padding[2]),'constant',0)[:,:,:,:,
                                                self.kernel_size[2]//2:self.kernel_size[2]//2+self.stride[2]*(conv3D_output_shape[2]-1)+1
                                                ], 
                                                weight=self.weight.unsqueeze(4), bias=None, 
                                                stride=self.stride,
                                                padding=(self.padding[0],self.padding[1],0),
                                                dilation=self.dilation,
                                                groups=self.groups)
        if self.mean:
            f = (f_a + f_c + f_s) / 3
        else:
            soft_w = self.soft_w_core.softmax(-1)
            f = f_a*soft_w[:,0].view(1,self.out_channels,1,1,1)+\
                f_c*soft_w[:,1].view(1,self.out_channels,1,1,1)+\
                f_s*soft_w[:,2].view(1,self.out_channels,1,1,1)

        if self.bias is not None:
            f += self.bias.view(1,self.out_channels,1,1,1)
            
        return f

    def extra_repr(self):
        s = super().extra_repr() + ', mean={mean}'
        return s.format(**self.__dict__)