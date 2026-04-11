import torch
import torch.nn as nn
import torch.nn.functional as F

def hard_swish(t):
    return t * F.relu6(t + 3.0) / 6.0

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt, sub):
        cb = conv_bias_opt if conv_bias_opt is not None else None
        y = F.conv_transpose3d(x, weight, cb, stride=1, padding=0, output_padding=0, dilation=1, groups=1)
        y = torch.logsumexp(y, dim=1, keepdim=True)
        y = hard_swish(y)
        y = y - sub.view(1, 1, 1, 1, 1)
        y = y.clamp(0.0, 1.0)
        return y.max(dim=2)[0]

def get_inputs():
    x = torch.rand(128, 3, 16, 32, 32)
    w = torch.rand(3, 16, 3, 3, 3)
    cb = torch.rand(16)
    su = torch.rand(1, 1, 1, 1)
    return [x, w, cb, su]

def get_init_inputs():
    return []
