import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Pybind: (x, w, b, scaling) with scaling scalar float tensor.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x, w, b, scaling):
        s = float(scaling.detach().cpu().item())
        y = F.linear(x, w, b)
        return torch.sigmoid(y) * s + y

batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0

def get_inputs():
    # Contract: x [1024,8192], w [8192,8192], b [8192], scaling shape [1]
    x = torch.rand(batch_size, input_size)
    w = torch.rand(hidden_size, input_size)
    b = torch.rand(hidden_size)
    scaling = torch.tensor([scaling_factor], dtype=torch.float32).contiguous()
    return [x, w, b, scaling]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
