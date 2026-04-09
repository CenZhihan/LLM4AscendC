import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            weight: Convolution weights

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return F.conv2d(x, weight, stride=1, padding=0, groups=self.in_channels)

# Test code
batch_size = 64
in_channels = 8
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    # weight shape for depthwise: (in_channels, 1, kernel_size, 1)
    weight = torch.rand(in_channels, 1, kernel_size, 1)
    return [x, weight]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, dilation]