import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt, residual):
        cb = conv_bias_opt if conv_bias_opt is not None else None
        y = F.conv_transpose3d(x, weight, cb, stride=1, padding=0, output_padding=0, dilation=1, groups=1)
        y = y + residual
        y = y * 2.0
        y = y + residual
        return y

def get_inputs():
    x = torch.rand(4, 32, 16, 16, 16)
    w = torch.rand(32, 64, 3, 3, 3)
    cb = torch.rand(64)
    r = torch.rand(4, 64, 18, 18, 18)
    return [x, w, cb, r]

def get_init_inputs():
    return []
