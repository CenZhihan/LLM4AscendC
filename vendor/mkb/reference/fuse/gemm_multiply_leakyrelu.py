import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Gemm, multiply, LeakyReLU.
    Pybind: 5x Tensor including scalar float32 tensors multiplier and negative_slope.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, w, b, multiplier, negative_slope):
        m = float(multiplier.detach().cpu().item())
        ns = float(negative_slope.detach().cpu().item())
        x = F.linear(x, w, b)
        x = x * m
        x = F.leaky_relu(x, negative_slope=ns)
        return x

batch_size = 1024
in_features  = 8192
out_features = 8192

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    b = torch.rand(out_features)
    multiplier = torch.tensor(2.0, dtype=torch.float32)
    negative_slope = torch.tensor(0.1, dtype=torch.float32)
    return [x, w, b, multiplier, negative_slope]

def get_init_inputs():
    return [in_features, out_features]
