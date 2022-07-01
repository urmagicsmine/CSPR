import torch.nn as nn

class Conv2_5d(nn.Conv3d):
    """
    Decorater class for Conv2_5d, in which kernel size is (1, K, K) or (K, 1, K) or (K, K, 1).

    Args:
        unsqueeze_axis: optional, the default axis is -3, resulting in a kernel size of (1, K, K)

        Other arguments are the same as torch.nn.Conv3d
    Examples:
        >>> import Conv2_5d
        >>> x = torch.rand(batch_size, 1, D, H, W)
        >>> # kernel size is (1, K, K)
        >>> conv = Conv2_5d(1, 64, 3, padding=1)
        >>> out = conv(x)

        >>> # kernel size is (K, K, 1)
        >>> conv = Conv2_5d(3, 64, 1, padding=1, unsqueeze_axis=-1)
        >>> out = conv(x)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode=None,
                        unsqueeze_axis=-3):
        self.unsqueeze_axis = unsqueeze_axis

        unsqueeze_axis += 3
        kernel_size = [kernel_size, kernel_size]
        padding = [padding, padding]
        kernel_size.insert(unsqueeze_axis, 1)
        padding.insert(unsqueeze_axis, 0)

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)