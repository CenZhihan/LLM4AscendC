import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, weight, bias optional, scale scalar tensor numel==1)."""
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Model, self).__init__()

    def forward(self, x, weight, bias_opt, scale):
        s = float(scale.reshape(-1)[0].item())
        conv_b = bias_opt if bias_opt is not None else None
        x = F.conv2d(x, weight, conv_b, stride=1, padding=0, dilation=1)
        x = x * s
        return torch.min(x, dim=1, keepdim=True)[0]

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    w = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    b = torch.rand(out_channels)
    sc = torch.tensor(scale_factor, dtype=torch.float32)
    return [x, w, b, sc]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
