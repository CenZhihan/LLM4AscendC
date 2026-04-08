import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, w, b, scaling) — scaling scalar float tensor."""
    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, w, b, scaling):
        s = float(scaling.detach().cpu().item())
        y = F.linear(x, w, b)
        return y * s + y

batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    b = torch.rand(out_features)
    scaling = torch.tensor(scaling_factor, dtype=torch.float32)
    return [x, w, b, scaling]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
