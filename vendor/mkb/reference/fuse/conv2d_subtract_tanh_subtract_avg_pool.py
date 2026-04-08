import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, weight, conv_bias optional)."""
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(Model, self).__init__()
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x, weight, conv_bias_opt):
        conv_b = conv_bias_opt if conv_bias_opt is not None else None
        x = F.conv2d(x, weight, conv_b, stride=1, padding=0, dilation=1)
        x = x - self.subtract1_value
        x = torch.tanh(x)
        x = x - self.subtract2_value
        return F.avg_pool2d(x, kernel_size=self.kernel_size_pool, stride=self.kernel_size_pool)

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    w = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    b = torch.rand(out_channels)
    return [x, w, b]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
