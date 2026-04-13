import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return F.conv_transpose3d(
            x, weight, None,
            stride=(2, 2, 2),
            padding=(1, 2, 3),
            output_padding=(1, 1, 1),
            groups=4,
            dilation=(1, 1, 1),
        )

def get_inputs():
    x = torch.rand(8, 32, 12, 24, 48)
    w = torch.rand(32, 8, 3, 5, 7)
    return [x, w]

def get_init_inputs():
    return []
