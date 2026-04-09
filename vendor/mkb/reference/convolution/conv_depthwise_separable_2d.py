import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w_depthwise, w_pointwise):
        x = F.conv2d(x, w_depthwise, None, stride=1, padding=1, groups=64)
        return F.conv2d(x, w_pointwise, None, stride=1, padding=0, groups=1)

def get_inputs():
    x = torch.rand(16, 64, 512, 512)
    w_dw = torch.rand(64, 1, 3, 3)
    w_pw = torch.rand(128, 64, 1, 1)
    return [x, w_dw, w_pw]

def get_init_inputs():
    return []
