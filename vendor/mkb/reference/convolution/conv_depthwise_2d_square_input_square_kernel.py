import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return F.conv2d(
            x, weight, None,
            stride=(1,1),
            padding=(0,0),
            dilation=(1,1),
            groups=64,
        )

def get_inputs():
    x = torch.rand(16, 64, 512, 512)
    w = torch.rand(64, 1, 3, 3)
    return [x, w]

def get_init_inputs():
    return []
