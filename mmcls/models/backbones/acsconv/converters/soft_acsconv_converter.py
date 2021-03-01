import torch.nn as nn
from .base_converter import BaseConverter
from ..operators import SoftACSConv

class SoftACSConverter(BaseConverter):
    """
    Decorator class for converting 2d convolution modules
    to corresponding soft-acs version in any networks.
    
    Args:
        model (torch.nn.module): model that needs to be converted
    Warnings:
        Functions in torch.nn.functional involved in data dimension are not supported
    Examples:
        >>> import SoftACSConverter
        >>> import torchvision
        >>> # m is a standard pytorch model
        >>> m = torchvision.models.resnet18(True)
        >>> m = SoftACSConverter(m)
        >>> # after converted, m is using SoftACSConv and capable of processing 3D volumes
        >>> x = torch.rand(batch_size, in_channels, D, H, W)
        >>> out = m(x)
    """
    converter_attributes = ['model', 'mean']
    target_conv = SoftACSConv

    def __init__(self, model, mean=False):
        preserve_state_dict = model.state_dict()
        self.mean = mean 
        model = self.convert_module(model)
        model.load_state_dict(preserve_state_dict,strict=False) # 
        self.model = model


    def convert_conv_kwargs(self, kwargs):
        kwargs['bias'] = True if kwargs['bias'] is not None else False
        kwargs['mean'] = self.mean
        return kwargs