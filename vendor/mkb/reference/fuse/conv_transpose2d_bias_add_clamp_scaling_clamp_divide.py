import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt, bias):
        cb = conv_bias_opt if conv_bias_opt is not None else None
        y = F.conv_transpose2d(
            x, weight, cb, stride=2, padding=1, output_padding=1, dilation=1, groups=1
        )
        y = y + bias.unsqueeze(0)
        y = y.clamp(0.0, 1.0)
        y = y * 2.0
        return (y / (y + 1e-6)).clamp(0.0, 1.0)

def get_inputs():
    x = torch.rand(128, 64, 128, 128)
    w = torch.rand(64, 64, 3, 3)
    cb = torch.rand(64)
    b = torch.rand(64, 1, 1)
    return [x, w, cb, b]

def get_init_inputs():
    return []
