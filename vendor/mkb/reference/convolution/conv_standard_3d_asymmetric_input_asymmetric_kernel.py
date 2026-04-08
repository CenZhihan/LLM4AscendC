import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return F.conv3d(
            x, weight, None,
            stride=(1,1,1),
            padding=(0,0,0),
            dilation=(1, 1, 1),
            groups=1,
        )

def get_inputs():
    x = torch.rand(8, 3, 16, 128, 128)
    w = torch.rand(64, 3, 3, 5, 7)
    return [x, w]

def get_init_inputs():
    return []
