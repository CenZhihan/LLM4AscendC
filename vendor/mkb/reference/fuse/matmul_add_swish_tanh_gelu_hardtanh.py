import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, w, b, add_value)."""
    def __init__(self, in_features, out_features, add_value_shape):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, w, b, add_value):
        x = F.linear(x, w, b)
        x = x + add_value
        x = torch.sigmoid(x) * x
        x = torch.tanh(x)
        x = F.gelu(x)
        return F.hardtanh(x, min_val=-1.0, max_val=1.0)

batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    b = torch.rand(out_features)
    add_value = torch.rand(add_value_shape)
    return [x, w, b, add_value]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]
