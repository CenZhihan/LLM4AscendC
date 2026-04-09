import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Pybind: (x, w, b, scaling, hardtanh_min, hardtanh_max) — scalar float tensors.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, w, b, scaling, hardtanh_min, hardtanh_max):
        s = float(scaling.detach().cpu().item())
        mn = float(hardtanh_min.detach().cpu().item())
        mx = float(hardtanh_max.detach().cpu().item())
        x = F.linear(x, w, b)
        x = x * s
        x = F.hardtanh(x, min_val=mn, max_val=mx)
        return F.gelu(x)

batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    b = torch.rand(out_features)
    scaling = torch.tensor(scaling_factor, dtype=torch.float32)
    ht_min = torch.tensor(float(hardtanh_min), dtype=torch.float32)
    ht_max = torch.tensor(float(hardtanh_max), dtype=torch.float32)
    return [x, w, b, scaling, ht_min, ht_max]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]
