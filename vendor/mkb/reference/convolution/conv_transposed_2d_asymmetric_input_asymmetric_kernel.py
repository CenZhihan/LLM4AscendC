import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return F.conv_transpose2d(
            x, weight, None,
            stride=(1,1),
            padding=(1,1),
            output_padding=(1,1),
            groups=1,
            dilation=(1,1),
        )

def get_inputs():
    x = torch.rand(64, 64, 128, 256)
    w = torch.rand(64, 64, 3, 5)
    return [x, w]

def get_init_inputs():
    return []
