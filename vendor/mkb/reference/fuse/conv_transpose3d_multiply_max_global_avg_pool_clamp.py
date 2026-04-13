import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt):
        cb = conv_bias_opt if conv_bias_opt is not None else None
        y = F.conv_transpose3d(x, weight, cb, stride=1, padding=0, output_padding=0, dilation=1, groups=1)
        y = y * y.max()
        y = F.adaptive_avg_pool3d(y, 1)
        return y.clamp(0.0, 1.0)

def get_inputs():
    x = torch.rand(128, 3, 16, 32, 32)
    w = torch.rand(3, 16, 3, 3, 3)
    cb = torch.rand(16)
    return [x, w, cb]

def get_init_inputs():
    return []
