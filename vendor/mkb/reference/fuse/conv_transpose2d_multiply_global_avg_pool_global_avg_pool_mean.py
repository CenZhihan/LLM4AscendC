import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias_opt):
        cb = conv_bias_opt
        y = F.conv_transpose2d(
            x, weight, cb, stride=2, padding=1, output_padding=1, dilation=1, groups=1,
        )
        y = y * 0.5
        y = F.adaptive_avg_pool2d(y, 1)
        y = F.adaptive_avg_pool2d(y, 1)
        return y.mean(dim=(2, 3), keepdim=True)

def get_inputs():
    # Contract: N=16, Cin=64, Cout=128, H=W=128; convT stride=2 pad=1 out_pad=1
    x = torch.rand(16, 64, 128, 128)
    w = torch.rand(64, 128, 3, 3)
    cb = torch.rand(128)
    return [x, w, cb]

def get_init_inputs():
    return []
