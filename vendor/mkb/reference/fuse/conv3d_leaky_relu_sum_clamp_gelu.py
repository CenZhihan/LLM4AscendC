import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias_opt, sum_tensor):
        b = bias_opt if bias_opt is not None else None
        y = F.conv3d(x, weight, b, stride=1, padding=0, dilation=1, groups=1)
        y = F.leaky_relu(y, 0.01)
        sf = sum_tensor.view(1, -1, 1, 1, 1)
        y = y + sf
        y = y.clamp(-1e6, 1e6)
        return F.gelu(y)

def get_inputs():
    x = torch.rand(128, 8, 16, 64, 64)
    w = torch.rand(64, 8, 3, 3, 3)
    cb = torch.rand(64)
    st = torch.rand(64, 1, 1, 1)
    return [x, w, cb, st]

def get_init_inputs():
    return []
