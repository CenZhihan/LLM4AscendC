import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

    def forward(self, x, w, linear_bias, add_bias, gamma, beta, num_groups, eps):
        """
        Args:
            x: Input tensor [M, K]
            w: Linear weight [N, K]
            linear_bias: Linear bias [N]
            add_bias: Bias to add after swish [N]
            gamma: GroupNorm weight [N]
            beta: GroupNorm bias [N]
            num_groups: Scalar int32 tensor (numel==1), same contract as custom op pybind
            eps: Scalar float32 tensor (numel==1)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        ng = int(num_groups.detach().cpu().item())
        ep = float(eps.detach().cpu().item())
        # matmul: x @ w^T + linear_bias
        x = F.linear(x, w, linear_bias)
        # swish: x * sigmoid(x)
        x = torch.sigmoid(x) * x
        # add bias
        x = x + add_bias
        # group norm - reshape to [batch, channel, 1] for 1D group norm
        x = x.unsqueeze(-1)  # [M, N] -> [M, N, 1]
        x = F.group_norm(x, ng, gamma, beta, ep)
        x = x.squeeze(-1)  # [M, N, 1] -> [M, N]
        return x

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)  # [N, K]
    linear_bias = torch.rand(out_features)
    add_bias = torch.rand(out_features)
    gamma = torch.rand(out_features)
    beta = torch.rand(out_features)
    # Match pybind: scalar tensors int32 / float32 (eval moves them to NPU with other inputs)
    num_groups_t = torch.tensor(64, dtype=torch.int32)
    eps_t = torch.tensor(1e-5, dtype=torch.float32)
    return [x, w, linear_bias, add_bias, gamma, beta, num_groups_t, eps_t]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]