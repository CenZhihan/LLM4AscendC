import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt, bias):
        cb = conv_bias_opt if conv_bias_opt is not None else None
        y = F.conv_transpose2d(x, weight, cb, stride=1, padding=0, output_padding=0, dilation=1, groups=1)
        y = F.adaptive_avg_pool2d(y, 1)
        y = y + bias.unsqueeze(0)
        y = torch.logsumexp(y.flatten(1), dim=1).sum().view(1, 1, 1, 1)  # scalar
        return y * y.sum()

def get_inputs():
    x = torch.rand(16, 64, 512, 512)
    w = torch.rand(64, 128, 3, 3)
    cb = torch.rand(128)
    b = torch.rand(128, 1, 1)
    return [x, w, cb, b]

def get_init_inputs():
    return []
