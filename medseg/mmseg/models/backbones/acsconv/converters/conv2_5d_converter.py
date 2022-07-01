import torch.nn as nn
from .base_converter import BaseConverter
from ..operators import Conv2_5d
from ..utils import _pair_same

class Conv2_5dConverter(BaseConverter):
    """
    Decorator class for converting 2d convolution modules
    to corresponding 3d version in any networks.
    
    Args:
        model (torch.nn.module): model that needs to be converted
    Warnings:
        Functions in torch.nn.functional involved in data dimension are not supported
    Examples:
        >>> import Conv2_5DWrapper
        >>> import torchvision
        >>> # m is a standard pytorch model
        >>> m = torchvision.models.resnet18(True)
        >>> m = Conv2_5DWrapper(m)
        >>> # after converted, m is using ACSConv and capable of processing 3D volumes
        >>> x = torch.rand(batch_size, in_channels, D, H, W)
        >>> out = m(x)
    """
    converter_attributes = ['model', 'unsqueeze_axis']
    target_conv = Conv2_5d

    def __init__(self, model, unsqueeze_axis=-3):
        preserve_state_dict = model.state_dict()
        self.model = model
        self.unsqueeze_axis = unsqueeze_axis
        self.model = self.convert_module(self.model)
        self.load_state_dict(preserve_state_dict, strict=True)
        
    def convert_conv_kwargs(self, kwargs):
        kwargs['bias'] = True if kwargs['bias'] is not None else False
        for k in ['kernel_size','stride','padding','dilation']:
            kwargs[k] = _pair_same(kwargs[k])[0]
        kwargs['unsqueeze_axis'] = self.unsqueeze_axis
        return kwargs

    def load_state_dict(self, state_dict, strict=True, unsqueeze_axis=-3):
        load_state_dict_from_2d_to_2_5d(self.model, state_dict, strict=strict, unsqueeze_axis=unsqueeze_axis)

def load_state_dict_from_2d_to_2_5d(model_2_5d, state_dict_2d, strict=True, unsqueeze_axis=-3):
    for key in list(state_dict_2d.keys()):
        if state_dict_2d[key].dim()==4:
            state_dict_2d[key] = state_dict_2d[key].unsqueeze(unsqueeze_axis)
    model_2_5d.load_state_dict(state_dict_2d, strict=strict)