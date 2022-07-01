
from .conv2_5d_converter import Conv2_5dConverter
from .conv3d_converter import Conv3dConverter
from .acsconv_converter import ACSConverter
from .soft_acsconv_converter import SoftACSConverter

def ConvertModel(model, converter):

    if converter == 'ACS':
        convert_func = ACSConverter
    elif converter == 'I3D':
        convert_func = Conv3dConverter
    elif converter == '2.5D':
        convert_func = Conv2_5dConverter

    model = convert_func(model)
    return model
