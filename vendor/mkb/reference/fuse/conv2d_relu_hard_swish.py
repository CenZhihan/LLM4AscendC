import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Conv2d + ReLU + HardSwish. Pybind: (x, weight, conv_bias optional).
    Hyperparams match kernel: stride=1, pad=0, dil=1; shapes N=128,Cin=8,Cout=64,H=W=128,K=3.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()

    def forward(self, x, weight, conv_bias_opt):
        conv_b = conv_bias_opt if conv_bias_opt is not None else None
        x = F.conv2d(x, weight, conv_b, stride=1, padding=0, dilation=1)
        x = torch.relu(x)
        return x * torch.clamp((x + 3) / 6, 0, 1)

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    w = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    b = torch.rand(out_channels)
    return [x, w, b]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
