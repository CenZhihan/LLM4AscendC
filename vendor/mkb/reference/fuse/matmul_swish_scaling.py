import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, w, b, scaling) — scaling scalar tensor."""
    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, w, b, scaling):
        s = float(scaling.detach().cpu().item())
        x = F.linear(x, w, b)
        x = x * torch.sigmoid(x)
        return x * s

batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    b = torch.rand(out_features)
    scaling = torch.tensor(scaling_factor, dtype=torch.float32)
    return [x, w, b, scaling]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
