import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Pybind: (x, w, b, gamma, beta, mul_w, num_groups int32, eps float32) — last two scalar tensors.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, w, b, gamma, beta, mul_w, num_groups, eps):
        ng = int(num_groups.detach().cpu().item())
        ep = float(eps.detach().cpu().item())
        x = F.linear(x, w, b)
        x = F.group_norm(x, ng, gamma, beta, ep)
        x = x * torch.sigmoid(x)
        x = x * mul_w
        x = x * torch.sigmoid(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features,)

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    b = torch.rand(out_features)
    gamma = torch.rand(out_features)
    beta = torch.rand(out_features)
    mul_w = torch.rand(out_features)
    num_groups_t = torch.tensor(num_groups, dtype=torch.int32)
    eps_t = torch.tensor(1e-5, dtype=torch.float32)
    return [x, w, b, gamma, beta, mul_w, num_groups_t, eps_t]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]
