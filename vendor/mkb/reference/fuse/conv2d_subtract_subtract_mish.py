import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model that performs a convolution, subtracts two values, applies Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, x, weight, conv_bias=None):
        x = F.conv2d(x, weight, conv_bias, stride=1, padding=0)
        x = x - self.subtract_value_1
        x = x - self.subtract_value_2
        x = torch.nn.functional.mish(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    weight = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    conv_bias = torch.rand(out_channels) if out_channels > 1 else None
    return [x, weight, conv_bias]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]