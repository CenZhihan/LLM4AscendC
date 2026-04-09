import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Pybind: (x, weight, bias optional, divisor float). divisor must be 2.0 per kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(Model, self).__init__()

    def forward(self, x, weight, bias_opt, divisor: float):
        conv_b = bias_opt if bias_opt is not None else None
        x = F.conv2d(x, weight, conv_b, stride=1, padding=0, dilation=1)
        x = x / divisor
        return F.leaky_relu(x, negative_slope=0.01)

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2.0

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    w = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    b = torch.rand(out_channels)
    return [x, w, b, divisor]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]
