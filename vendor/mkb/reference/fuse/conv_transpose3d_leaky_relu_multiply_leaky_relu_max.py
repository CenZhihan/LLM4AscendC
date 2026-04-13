import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt, scale):
        cb = conv_bias_opt if conv_bias_opt is not None else None
        y = F.conv_transpose3d(x, weight, cb, stride=1, padding=0, output_padding=0, dilation=1, groups=1)
        y = F.leaky_relu(y, 0.01)
        y = y * scale.unsqueeze(0)
        y = F.leaky_relu(y, 0.01)
        return torch.max(y, dim=2)[0]

def get_inputs():
    x = torch.rand(16, 16, 16, 32, 32)
    w = torch.rand(16, 32, 3, 3, 3)
    cb = torch.rand(32)
    sc = torch.rand(32, 1, 1, 1)
    return [x, w, cb, sc]

def get_init_inputs():
    return []
