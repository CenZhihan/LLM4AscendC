import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt, scaling_factor, bias):
        b = conv_bias_opt if conv_bias_opt is not None else None
        y = F.conv3d(x, weight, b, stride=1, padding=0, dilation=1, groups=1)
        y = y * scaling_factor.unsqueeze(0) + bias.unsqueeze(0)
        y = torch.tanh(y)
        y = y * torch.sigmoid(y)
        return y

def get_inputs():
    x = torch.rand(128, 3, 16, 64, 64)
    w = torch.rand(16, 3, 3, 3, 3)
    cb = torch.rand(16)
    sc = torch.rand(16, 1, 1, 1)
    bi = torch.rand(16, 1, 1, 1)
    return [x, w, cb, sc, bi]

def get_init_inputs():
    return []
