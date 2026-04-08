import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    Pybind contract: leaky_relu_custom(Tensor x, float negative_slope).
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, negative_slope: float) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=negative_slope)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x, 0.01]

def get_init_inputs():
    return []
