import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt, scale1, bias, scale2):
        cb = conv_bias_opt if conv_bias_opt is not None else None
        y = F.conv_transpose3d(x, weight, cb, stride=1, padding=0, output_padding=0, dilation=1, groups=1)
        y = y * scale1.view(1, -1, 1, 1, 1)
        y = F.adaptive_avg_pool3d(y, 1)
        y = y + bias.view(1, -1, 1, 1, 1)
        return y * scale2.view(1, -1, 1, 1, 1)

def get_inputs():
    x = torch.rand(4, 32, 16, 16, 16)
    w = torch.rand(32, 64, 3, 3, 3)
    cb = torch.rand(64)
    s1 = torch.rand(64)
    b = torch.rand(64, 1, 1, 1)
    s2 = torch.rand(64)
    return [x, w, cb, s1, b, s2]

def get_init_inputs():
    return []
