import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Pybind: (x, w, bias, gamma, beta, eps, negative_slope) — eps and negative_slope scalar float tensors.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(Model, self).__init__()
        self.num_groups = num_groups

    def forward(self, x, w, bias, gamma, beta, eps, negative_slope):
        ep = float(eps.detach().cpu().item())
        ns = float(negative_slope.detach().cpu().item())
        x = F.linear(x, w, bias)
        x = F.group_norm(x, self.num_groups, gamma, beta, ep)
        x = F.leaky_relu(x, negative_slope=ns)
        return x + x

batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512

def get_inputs():
    x = torch.rand(batch_size, input_size)
    w = torch.rand(hidden_size, input_size)
    b = torch.rand(hidden_size)
    gamma = torch.rand(hidden_size)
    beta = torch.rand(hidden_size)
    eps_t = torch.tensor(1e-5, dtype=torch.float32)
    ns_t = torch.tensor(0.01, dtype=torch.float32)
    return [x, w, b, gamma, beta, eps_t, ns_t]

def get_init_inputs():
    return [input_size, hidden_size, num_groups]
