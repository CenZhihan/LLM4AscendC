import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_GROUPS = 8

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias, gamma, beta):
        y = F.conv3d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1)
        y = F.group_norm(y, NUM_GROUPS, gamma, beta, eps=1e-5)
        y = torch.min(y, dim=2)[0]
        y = y.clamp(0.0, 1.0)
        return F.dropout(y, p=0.0, training=False)

def get_inputs():
    x = torch.rand(16, 3, 16, 64, 64)
    w = torch.rand(16, 3, 3, 3, 3)
    b = torch.rand(16)
    gamma = torch.rand(16)
    bt = torch.rand(16)
    return [x, w, b, gamma, bt]

def get_init_inputs():
    return []
