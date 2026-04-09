import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias):
        y = F.conv3d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1)
        y = torch.min(y, dim=2)[0]
        return F.softmax(y, dim=1)

def get_inputs():
    x = torch.rand(128, 3, 24, 32, 32)
    w = torch.rand(24, 3, 3, 3, 3)
    b = torch.rand(24)
    return [x, w, b]

def get_init_inputs():
    return []
