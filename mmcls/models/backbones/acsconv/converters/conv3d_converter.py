import torch
import torch.nn as nn
from .base_converter import BaseConverter
from ..utils import _triple_same

class Conv3dConverter(BaseConverter):
    """
    Decorator class for converting 2d convolution modules
    to corresponding 3d version in any networks.

    Args:
        model (torch.nn.module): model that needs to be converted
    Warnings:
        Functions in torch.nn.functional involved in data dimension are not supported
    Examples:
        >>> import Conv3dConverter
        >>> import torchvision
        >>> # m is a standard pytorch model
        >>> m = torchvision.models.resnet18(True)
        >>> m = Conv3dConverter(m)
        >>> # after converted, m is using Conv3d and capable of processing 3D volumes
        >>> x = torch.rand(batch_size, in_channels, D, H, W)
        >>> out = m(x)
    """
    converter_attributes = ['model']
    target_conv = nn.Conv3d

    def __init__(self, model, i3d_repeat_axis=None):
        if i3d_repeat_axis is not None:
            preserve_state_dict = model.state_dict()
        self.model = model
        self.model = self.convert_module(self.model)
        if i3d_repeat_axis is not None:
            self.load_state_dict(preserve_state_dict, strict=True, i3d_repeat_axis=i3d_repeat_axis)

    def convert_conv_kwargs(self, kwargs):
        kwargs['bias'] = True if kwargs['bias'] is not None else False
        for k in ['kernel_size','stride','padding','dilation']:
            kwargs[k] = _triple_same(kwargs[k])
        return kwargs

    def load_state_dict(self, state_dict, strict=True, i3d_repeat_axis=None):
        if i3d_repeat_axis is not None:
            return load_state_dict_from_2d_to_i3d(self.model, state_dict, strict, repeat_axis=i3d_repeat_axis)
        else:
            return self.model.load_state_dict(state_dict, strict)


def load_state_dict_from_2d_to_i3d(model_3d, state_dict_2d, strict=True, repeat_axis=-1):
    present_dict = model_3d.state_dict()
    for key in list(state_dict_2d.keys()):
        if state_dict_2d[key].dim()==4:
            repeat_times = present_dict[key].shape[repeat_axis]
            state_dict_2d[key] = torch.stack([state_dict_2d[key]]*repeat_times, dim=repeat_axis) / repeat_times
    return model_3d.load_state_dict(state_dict_2d, strict=strict)
