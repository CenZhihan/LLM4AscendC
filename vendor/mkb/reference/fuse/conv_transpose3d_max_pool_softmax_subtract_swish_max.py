import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(t):
    return t * torch.sigmoid(t)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt, sub):
        cb = conv_bias_opt if conv_bias_opt is not None else None
        y = F.conv_transpose3d(x, weight, cb, stride=1, padding=0, output_padding=0, dilation=1, groups=1)
        y = F.max_pool3d(y, kernel_size=2, stride=2)
        y = F.softmax(y, dim=1)
        y = y - sub.view(1, -1, 1, 1, 1)
        y = swish(y)
        return y.max(dim=2)[0]

def get_inputs():
    x = torch.rand(4, 32, 16, 16, 16)
    w = torch.rand(32, 64, 3, 3, 3)
    cb = torch.rand(64)
    su = torch.rand(64)
    return [x, w, cb, su]

def get_init_inputs():
    return []
