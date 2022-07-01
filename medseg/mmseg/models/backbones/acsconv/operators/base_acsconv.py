import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import math
from collections import OrderedDict

from ..utils import _to_triple, _triple_same, _pair_same

class _ACSConv(nn.Module):
    """
    Base class for ACS Convolution
    Basically the same with _ConvNd in torch.nn.

    Warnings:
        The kernel size should be the same in the three directions under this implementation.
    """
    def __init__(self,  in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super().__init__()

        assert padding_mode!='circular', 'circular padding is not supported yet.'
        stride = _to_triple(stride)
        padding = _to_triple(padding)
        dilation = _to_triple(dilation)
        output_padding = _to_triple(output_padding)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        
        if self.transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *_pair_same(kernel_size) ))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *_pair_same(kernel_size) ))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _triple_same(kernel_size) 

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'
