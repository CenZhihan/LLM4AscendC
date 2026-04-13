import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt, bias):
        cb = conv_bias_opt if conv_bias_opt is not None else None
        r = F.conv_transpose3d(
            x, weight, cb, stride=1, padding=0, output_padding=0, dilation=1, groups=1
        )
        b = bias.unsqueeze(0)
        return ((2 * r + b) * r) + r

def get_inputs():
    x = torch.rand(16, 32, 16, 32, 32)
    w = torch.rand(32, 64, 3, 3, 3)
    cb = torch.rand(64)
    bias = torch.rand(64, 1, 1, 1)
    return [x, w, cb, bias]

def get_init_inputs():
    return []
