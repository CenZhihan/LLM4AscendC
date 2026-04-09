import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple model that performs an ELU activation.
    Pybind contract: elu_custom(Tensor x, float alpha).
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return F.elu(x, alpha=alpha)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x, 1.0]

def get_init_inputs():
    return []
