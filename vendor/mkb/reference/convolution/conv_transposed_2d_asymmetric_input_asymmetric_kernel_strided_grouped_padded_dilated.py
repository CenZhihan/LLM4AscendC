import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return F.conv_transpose2d(
            x, weight, None,
            stride=(2, 3),
            padding=(1, 2),
            output_padding=(0, 0),
            groups=4,
            dilation=(2, 1),
        )

def get_inputs():
    x = torch.rand(16, 32, 128, 256)
    w = torch.rand(32, 16, 3, 5)
    return [x, w]

def get_init_inputs():
    return []
