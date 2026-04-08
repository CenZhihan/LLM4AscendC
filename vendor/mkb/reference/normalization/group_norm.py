import torch
import torch.nn as nn
import torch.nn.functional as F

# Pybind matches kernel: group_norm_custom(x, gamma, beta) with baked num_groups=8, eps=1e-5
NUM_GROUPS = 8

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        return F.group_norm(x, NUM_GROUPS, gamma, beta, eps=1e-5)

batch_size = 128
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    gamma = torch.rand(features)
    beta = torch.rand(features)
    return [x, gamma, beta]

def get_init_inputs():
    return []
