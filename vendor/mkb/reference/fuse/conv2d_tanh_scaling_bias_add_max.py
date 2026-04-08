import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Pybind: (x, weight, conv_bias optional, post_bias [Cout,1,1] or [Cout], scaling_factor scalar tensor).
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(Model, self).__init__()
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x, weight, conv_bias_opt, post_bias, scaling_factor):
        s = float(scaling_factor.reshape(-1)[0].item())
        conv_b = conv_bias_opt if conv_bias_opt is not None else None
        x = F.conv2d(x, weight, conv_b, stride=1, padding=0, dilation=1)
        x = torch.tanh(x)
        x = x * s
        if post_bias.dim() == 1:
            x = x + post_bias.view(1, -1, 1, 1)
        else:
            x = x + post_bias
        return F.max_pool2d(x, kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size)

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 4

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    w = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    cb = torch.rand(out_channels)
    pb = torch.randn(bias_shape)
    sc = torch.tensor(scaling_factor, dtype=torch.float32)
    return [x, w, cb, pb, sc]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]
