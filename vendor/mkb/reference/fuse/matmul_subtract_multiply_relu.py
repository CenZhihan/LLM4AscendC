import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, w, b, sub, mul) — sub/mul scalar float tensors."""
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, w, b, sub, mul):
        sv = float(sub.detach().cpu().item())
        mv = float(mul.detach().cpu().item())
        x = F.linear(x, w, b)
        x = x - sv
        x = x * mv
        return torch.relu(x)

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    b = torch.rand(out_features)
    sub = torch.tensor(subtract_value, dtype=torch.float32)
    mul = torch.tensor(multiply_value, dtype=torch.float32)
    return [x, w, b, sub, mul]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]
