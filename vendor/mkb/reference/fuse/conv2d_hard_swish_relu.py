import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model that performs a convolution, applies HardSwish, and then ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x, weight, conv_bias=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            weight: Convolution weights
            conv_bias: Optional convolution bias

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = F.conv2d(x, weight, conv_bias, stride=1, padding=0)
        x = torch.nn.functional.hardswish(x)
        x = torch.relu(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    weight = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    conv_bias = torch.rand(out_channels) if out_channels > 1 else None
    return [x, weight, conv_bias]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]