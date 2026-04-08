import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Pybind: (x, weight, conv_bias optional).
    Conv stride=1 pad=0; AvgPool k=4 s=4 pad=0; sigmoid; sum over CHW -> [N].
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()

    def forward(self, x, weight, conv_bias_opt):
        conv_b = conv_bias_opt if conv_bias_opt is not None else None
        x = F.conv2d(x, weight, conv_b, stride=1, padding=0, dilation=1)
        x = F.avg_pool2d(x, kernel_size=4, stride=4, padding=0)
        x = torch.sigmoid(x)
        return torch.sum(x, dim=[1, 2, 3])

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 384, 384
kernel_size = 3
pool_kernel_size = 4

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    w = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    b = torch.rand(out_channels)
    return [x, w, b]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
