import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, weight, conv_bias optional, post_bias [Cout,1,1])."""
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()

    def forward(self, x, weight, conv_bias_opt, bias):
        conv_b = conv_bias_opt if conv_bias_opt is not None else None
        x = F.conv2d(x, weight, conv_b, stride=1, padding=0, dilation=1)
        x = torch.relu(x)
        return x + bias

batch_size = 128
in_channels  = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    w = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    cb = torch.rand(out_channels)
    pb = torch.randn(bias_shape)
    return [x, w, cb, pb]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
