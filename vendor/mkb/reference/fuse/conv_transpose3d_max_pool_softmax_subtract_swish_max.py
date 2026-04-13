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
        y = F.conv_transpose3d(
            x, weight, cb, stride=2, padding=1, output_padding=1, dilation=1, groups=1,
        )
        y = F.max_pool3d(y, kernel_size=2, stride=2)
        y = F.softmax(y, dim=1)
        y = y - sub.view(1, -1, 1, 1, 1)
        y = swish(y)
        return y.max(dim=2)[0]

def get_inputs():
    # Contract: N=128, Cin=3, Cout=16, D/H/W=16/32/32; subtract [Cout]
    x = torch.rand(128, 3, 16, 32, 32)
    w = torch.rand(3, 16, 3, 3, 3)
    cb = torch.rand(16)
    su = torch.rand(16)
    return [x, w, cb, su]

def get_init_inputs():
    return []
