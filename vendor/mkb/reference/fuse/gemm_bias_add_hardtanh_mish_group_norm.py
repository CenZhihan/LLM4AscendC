import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Pybind: (x, weight, lin_bias, bias, gn_gamma, gn_beta).
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(Model, self).__init__()
        self.num_groups = num_groups

    def forward(self, x, weight, lin_bias, bias, gn_gamma, gn_beta):
        x = F.linear(x, weight, lin_bias)
        x = x + bias
        x = F.hardtanh(x)
        x = F.mish(x)
        x = F.group_norm(x, self.num_groups, gn_gamma, gn_beta)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)
num_groups = 256

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    lin_bias = torch.rand(out_features)
    bias = torch.rand(out_features)
    gn_gamma = torch.rand(out_features)
    gn_beta = torch.rand(out_features)
    return [x, w, lin_bias, bias, gn_gamma, gn_beta]

def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]
