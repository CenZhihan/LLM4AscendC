import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt, bias):
        cb = conv_bias_opt if conv_bias_opt is not None else None
        y = F.conv_transpose2d(x, weight, cb, stride=2, padding=1, output_padding=1, dilation=1, groups=1)
        y = F.softmax(y, dim=1)
        y = y + bias.unsqueeze(0)
        y = y * 2.0
        return torch.sigmoid(y)

def get_inputs():
    x = torch.rand(128, 64, 32, 32)
    w = torch.rand(64, 128, 4, 4)
    cb = torch.rand(128)
    b = torch.rand(128, 1, 1)
    return [x, w, cb, b]

def get_init_inputs():
    return []
