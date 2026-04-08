import torch
import torch.nn as nn
import torch.nn.functional as F

# Cout=24, num_groups=8
NUM_GROUPS = 8

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias, gamma, beta):
        y = F.conv3d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1)
        y = F.group_norm(y, NUM_GROUPS, gamma, beta, eps=1e-5)
        return y.mean(dim=(1, 2, 3, 4))

def get_inputs():
    x = torch.rand(16, 3, 24, 32, 32)
    w = torch.rand(24, 3, 3, 3, 3)
    b = torch.rand(24)
    gamma = torch.rand(24)
    bt = torch.rand(24)
    return [x, w, b, gamma, bt]

def get_init_inputs():
    return []
