import torch
import torch.nn as nn
import torch.nn.functional as F

def mish(t):
    return t * torch.tanh(F.softplus(t))

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt):
        cb = conv_bias_opt if conv_bias_opt is not None else None
        y = F.conv_transpose2d(x, weight, cb, stride=1, padding=0, output_padding=0, dilation=1, groups=1)
        y = mish(y + 1.0)
        return F.hardtanh(y * 1.0, -1.0, 1.0)

def get_inputs():
    x = torch.rand(128, 64, 128, 128)
    w = torch.rand(64, 64, 3, 3)
    cb = torch.rand(64)
    return [x, w, cb]

def get_init_inputs():
    return []
