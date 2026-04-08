import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, weight, lin_bias, gn_gamma, gn_beta)."""
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x, weight, lin_bias, gn_gamma, gn_beta):
        x = F.linear(x, weight, lin_bias)
        x = F.group_norm(x, self.num_groups, gn_gamma, gn_beta)
        return F.hardtanh(x, min_val=self.hardtanh_min, max_val=self.hardtanh_max)

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 16
hardtanh_min = -2.0
hardtanh_max = 2.0

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    lin_bias = torch.rand(out_features)
    gn_gamma = torch.rand(out_features)
    gn_beta = torch.rand(out_features)
    return [x, w, lin_bias, gn_gamma, gn_beta]

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]
