import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, w, b, divisor) — divisor scalar tensor."""
    def __init__(self, input_size, output_size, divisor):
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x, w, b, divisor):
        d = float(divisor.detach().cpu().item())
        x = F.linear(x, w, b)
        x = x / d
        return F.gelu(x)

batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0

def get_inputs():
    x = torch.rand(batch_size, input_size)
    w = torch.rand(output_size, input_size)
    b = torch.rand(output_size)
    divisor_t = torch.tensor(divisor, dtype=torch.float32)
    return [x, w, b, divisor_t]

def get_init_inputs():
    return [input_size, output_size, divisor]
