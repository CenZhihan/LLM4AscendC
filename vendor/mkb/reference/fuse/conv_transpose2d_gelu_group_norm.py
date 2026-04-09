import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_GROUPS = 8

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        y = F.gelu(x)
        return F.group_norm(y, NUM_GROUPS, gamma, beta, eps=1e-5)

def get_inputs():
    x = torch.rand(4, 64, 32, 32)
    gamma = torch.rand(64)
    bt = torch.rand(64)
    return [x, gamma, bt]

def get_init_inputs():
    return []
