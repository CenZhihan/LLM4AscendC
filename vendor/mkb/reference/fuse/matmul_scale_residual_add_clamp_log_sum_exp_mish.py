import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Pybind: (x, weight, bias, scaling, clamp_min, clamp_max) — last three length-1 float tensors.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x, weight, bias, scaling, clamp_min, clamp_max):
        s = float(scaling.reshape(-1)[0].detach().cpu().item())
        lo = float(clamp_min.reshape(-1)[0].detach().cpu().item())
        hi = float(clamp_max.reshape(-1)[0].detach().cpu().item())
        x = F.linear(x, weight, bias)
        x = x * s + x
        x = torch.clamp(x, lo, hi)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        return x * F.mish(x)

batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

def get_inputs():
    x = torch.rand(batch_size, input_size)
    w = torch.rand(hidden_size, input_size)
    b = torch.rand(hidden_size)
    scaling = torch.tensor([scale_factor], dtype=torch.float32)
    cmin = torch.tensor([clamp_min], dtype=torch.float32)
    cmax = torch.tensor([clamp_max], dtype=torch.float32)
    return [x, w, b, scaling, cmin, cmax]

def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]
