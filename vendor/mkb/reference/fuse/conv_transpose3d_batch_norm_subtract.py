import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, conv_bias, bn_weight, bn_bias):
        y = F.conv_transpose3d(
            x, weight, conv_bias, stride=2, padding=1, output_padding=0, dilation=1, groups=1,
        )
        c = y.size(1)
        rm = torch.zeros(c, device=y.device, dtype=y.dtype)
        rv = torch.ones(c, device=y.device, dtype=y.dtype)
        y = F.batch_norm(y, rm, rv, bn_weight, bn_bias, training=False, momentum=0.0)
        return y - y.mean(dim=(2, 3, 4), keepdim=True)

def get_inputs():
    # Contract: Cin=16, Cout=32, spatial [16,32,32], weight [16,32,3,3,3]; bn length 32
    x = torch.rand(4, 16, 16, 32, 32)
    w = torch.rand(16, 32, 3, 3, 3)
    cb = torch.rand(32)
    bw = torch.rand(32)
    bb = torch.rand(32)
    return [x, w, cb, bw, bb]

def get_init_inputs():
    return []
