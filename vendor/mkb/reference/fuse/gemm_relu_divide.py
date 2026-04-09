import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Pybind: (x, w, b, divisor) with divisor scalar float tensor on NPU.
    """
    def __init__(self, in_features, out_features, divisor):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, w, b, divisor):
        d = float(divisor.detach().cpu().item())
        x = F.linear(x, w, b)
        x = torch.relu(x)
        return x / d

batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    b = torch.rand(out_features)
    divisor_t = torch.tensor(divisor, dtype=torch.float32)
    return [x, w, b, divisor_t]

def get_init_inputs():
    return [in_features, out_features, divisor]
