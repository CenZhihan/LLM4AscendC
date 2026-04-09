import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Gemm + bias + ReLU. Pybind: (x, w, bias).
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, w, bias):
        x = F.linear(x, w, bias)
        return torch.relu(x)

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    bias = torch.rand(out_features)
    return [x, w, bias]

def get_init_inputs():
    return [in_features, out_features, bias_shape]
