import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, weight, bias optional, multiplier tensor [Cout,1,1] or broadcast)."""
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(Model, self).__init__()

    def forward(self, x, weight, bias_opt, multiplier):
        conv_b = bias_opt if bias_opt is not None else None
        x = F.conv2d(x, weight, conv_b, stride=1, padding=0, dilation=1)
        x = x * multiplier
        x = F.leaky_relu(x, negative_slope=0.01)
        return F.gelu(x)

batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    w = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    b = torch.rand(out_channels)
    m = torch.randn(multiplier_shape)
    return [x, w, b, m]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]
