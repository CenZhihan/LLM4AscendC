import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, w, b)."""
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, w, b):
        x = F.linear(x, w, b)
        x = x * torch.sigmoid(x)
        x = x / 2.0
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = torch.tanh(x)
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x

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
