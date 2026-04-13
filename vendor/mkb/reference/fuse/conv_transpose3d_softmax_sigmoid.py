import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias):
        y = F.conv_transpose3d(x, weight, bias, stride=1, padding=0, output_padding=0, dilation=1, groups=1)
        y = F.softmax(y, dim=1)
        return torch.sigmoid(y)

def get_inputs():
    x = torch.rand(16, 32, 16, 32, 32)
    w = torch.rand(32, 64, 3, 3, 3)
    b = torch.rand(64)
    return [x, w, b]

def get_init_inputs():
    return []
