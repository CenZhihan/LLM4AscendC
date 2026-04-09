import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, w, b)."""
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, w, b):
        x = F.linear(x, w, b)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        return torch.logsumexp(x, dim=1, keepdim=True)

batch_size = 1024
in_features  = 8192
out_features = 8192

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    b = torch.rand(out_features)
    return [x, w, b]

def get_init_inputs():
    return [in_features, out_features]
