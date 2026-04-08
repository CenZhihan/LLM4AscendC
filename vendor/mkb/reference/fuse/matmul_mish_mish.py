import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, weight, bias)."""
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, weight, bias):
        x = F.linear(x, weight, bias)
        x = F.mish(x)
        return F.mish(x)

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    b = torch.rand(out_features)
    return [x, w, b]

def get_init_inputs():
    return [in_features, out_features]
