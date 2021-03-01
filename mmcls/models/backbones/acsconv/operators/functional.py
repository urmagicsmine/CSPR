import torch
import torch.nn.functional as F
import math
import os


def conv3D_output_shape_f(i, input_shape, kernel_size, dilation, padding, stride):
    """
    Calculate the original output size assuming the convolution is nn.Conv3d based on 
    input size, kernel size, dilation, padding and stride.
    """
    return math.floor((input_shape[i]-kernel_size[i]-(dilation[i]-1)*
                                        (kernel_size[i]-1)+2*padding[i])
                                    /stride[i])+1
    

def acs_conv_f(x, weight, bias, kernel_size, dilation, padding, stride, groups, out_channels, acs_kernel_split):
    B, C_in, *input_shape = x.shape
    conv3D_output_shape = (conv3D_output_shape_f(0, input_shape, kernel_size, dilation, padding, stride), 
                            conv3D_output_shape_f(1, input_shape, kernel_size, dilation, padding, stride), 
                            conv3D_output_shape_f(2, input_shape, kernel_size, dilation, padding, stride))
            
    weight_a = weight[0:acs_kernel_split[0]].unsqueeze(2)
    weight_c = weight[acs_kernel_split[0]:(acs_kernel_split[0]+acs_kernel_split[1])].unsqueeze(3)
    weight_s = weight[(acs_kernel_split[0]+acs_kernel_split[1]):].unsqueeze(4)
    f_out = []
    if acs_kernel_split[0]>0:
        a = F.conv3d(x if conv3D_output_shape[0]==input_shape[0] or 2*conv3D_output_shape[0]==input_shape[0] else F.pad(x, (0,0,0,0,padding[0],padding[0]),'constant',0)[:,:,
                                            kernel_size[0]//2:kernel_size[0]//2+(conv3D_output_shape[0]-1)*stride[0]+1,
                                            :,:], 
                                            weight=weight_a, bias=None, 
                                            stride=stride,
                                            padding=(0,padding[1],padding[2]),
                                            dilation=dilation,
                                            groups=groups)                
        f_out.append(a)
    if acs_kernel_split[1]>0:
        c = F.conv3d(x if conv3D_output_shape[1]==input_shape[1] or 2*conv3D_output_shape[1]==input_shape[1] else F.pad(x, (0,0,padding[1],padding[1]),'constant',0)[:,:,:,
                                            kernel_size[1]//2:kernel_size[1]//2+stride[1]*(conv3D_output_shape[1]-1)+1,
                                            :], 
                                            weight=weight_c, bias=None,                                     
                                            stride=stride,
                                            padding=(padding[0],0,padding[2]),
                                            dilation=dilation,
                                            groups=groups)
        f_out.append(c)
    if acs_kernel_split[2]>0:
        s = F.conv3d(x if conv3D_output_shape[2]==input_shape[2] or 2*conv3D_output_shape[2]==input_shape[2] else F.pad(x, (padding[2],padding[2]),'constant',0)[:,:,:,:,
                                            kernel_size[2]//2:kernel_size[2]//2+stride[2]*(conv3D_output_shape[2]-1)+1
                                            ], 
                                            weight=weight_s, 
                                            bias=None, 
                                            stride=stride,
                                            padding=(padding[0],padding[1],0),
                                            dilation=dilation,
                                            groups=groups)
        f_out.append(s)
    f = torch.cat(f_out, dim=1)
    if bias is not None:
        f += bias.view(1,out_channels,1,1,1)
    return f
