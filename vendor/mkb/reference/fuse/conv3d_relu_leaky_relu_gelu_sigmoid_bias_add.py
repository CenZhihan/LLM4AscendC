import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt, bias):
        b = conv_bias_opt if conv_bias_opt is not None else None
        y = F.conv3d(x, weight, b, stride=1, padding=0, dilation=1, groups=1)
        y = F.relu(y)
        y = F.leaky_relu(y, 0.01)
        y = F.gelu(y)
        y = torch.sigmoid(y)
        return y + bias.view(1, -1, 1, 1, 1)

def get_inputs():
    x = torch.rand(128, 8, 16, 64, 64)
    w = torch.rand(64, 8, 3, 3, 3)
    cb = torch.rand(64)
    pb = torch.rand(64, 1, 1, 1)
    return [x, w, cb, pb]

def get_init_inputs():
    return []
