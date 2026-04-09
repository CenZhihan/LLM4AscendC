import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Pybind: (x, weight, conv_bias, gn_gamma, gn_beta, scale_1d) — all tensors; bias 1D [Cout].
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x, weight, bias, gn_gamma, gn_beta, scale_1d):
        x = F.conv2d(x, weight, bias, stride=1, padding=0, dilation=1)
        x = F.group_norm(x, self.num_groups, gn_gamma, gn_beta)
        x = x * scale_1d.view(1, -1, 1, 1)
        x = F.max_pool2d(x, kernel_size=4, stride=4, padding=0)
        return torch.clamp(x, self.clamp_min, self.clamp_max)

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
num_groups = 16
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 4
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    w = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    b = torch.rand(out_channels)
    g = torch.rand(out_channels)
    beta = torch.rand(out_channels)
    sc = torch.rand(out_channels)
    return [x, w, b, g, beta, sc]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]
