import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, w, b)."""
    def __init__(self, in_features, out_features, max_dim):
        super(Model, self).__init__()
        self.max_dim = max_dim

    def forward(self, x, w, b):
        x = F.linear(x, w, b)
        x = torch.max(x, dim=self.max_dim, keepdim=True).values
        x = x - x.mean(dim=1, keepdim=True)
        return F.gelu(x)

batch_size = 1024
in_features = 8192
out_features = 8192
max_dim = 1

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    b = torch.rand(out_features)
    return [x, w, b]

def get_init_inputs():
    return [in_features, out_features, max_dim]
