import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, weight) 1D conv; sizes from kernelbench txt."""
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return F.conv1d(
            x, weight, None,
            stride=1, padding=0, dilation=1, groups=1,
        )

def get_inputs():
    x = torch.rand(32, 64, 131072)
    w = torch.rand(128, 64, 3)
    return [x, w]

def get_init_inputs():
    return []
