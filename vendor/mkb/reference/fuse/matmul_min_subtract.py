import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, w, b, c) — c scalar float tensor."""
    def __init__(self, in_features, out_features, constant):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, w, b, c):
        cv = float(c.detach().cpu().item())
        x = F.linear(x, w, b)
        cap = torch.as_tensor(cv, device=x.device, dtype=x.dtype)
        x = torch.minimum(x, cap)
        return x - cv

batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    b = torch.rand(out_features)
    c = torch.tensor(constant, dtype=torch.float32)
    return [x, w, b, c]

def get_init_inputs():
    return [in_features, out_features, constant]
