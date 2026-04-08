import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt, bias):
        cb = conv_bias_opt if conv_bias_opt is not None else None
        y = F.conv_transpose2d(x, weight, cb, stride=1, padding=0, output_padding=0, dilation=1, groups=1)
        y = y + bias.view(1, -1, 1, 1)
        y = y.clamp(0.0, 1.0)
        y = y * 2.0
        return (y / (y + 1e-6)).clamp(0.0, 1.0)

def get_inputs():
    x = torch.rand(8, 64, 32, 32)
    w = torch.rand(64, 32, 3, 3)
    cb = torch.rand(32)
    b = torch.rand(32, 1, 1)
    return [x, w, cb, b]

def get_init_inputs():
    return []
