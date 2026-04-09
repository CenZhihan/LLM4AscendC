import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, weight, conv_bias optional, bias [Cout,1,1,1]) -> [N,1,1,1] fused."""

    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt, bias):
        cb = conv_bias_opt if conv_bias_opt is not None else None
        y = F.conv3d(x, weight, cb, stride=1, padding=0, dilation=1, groups=1)
        y = y / (y.flatten(1).abs().max(dim=-1, keepdim=True)[0].view(-1, 1, 1, 1, 1) + 1e-8)
        y = F.adaptive_avg_pool3d(y, 1)
        y = y + bias.view(1, -1, 1, 1, 1)
        return y.sum(dim=1, keepdim=True)

def get_inputs():
    x = torch.rand(128, 8, 16, 64, 64)
    w = torch.rand(16, 8, 3, 3, 3)
    cb = torch.rand(16)
    b = torch.rand(16, 1, 1, 1)
    return [x, w, cb, b]

def get_init_inputs():
    return []
