import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """SqueezeNet fire: squeeze 1x1 -> expand 1x1 + 3x3, concat, ReLU."""

    def __init__(self):
        super().__init__()

    def forward(self, x, w_squeeze, b_squeeze, w_expand1, b_expand1, w_expand3, b_expand3):
        s = F.relu(F.conv2d(x, w_squeeze, b_squeeze, stride=1, padding=0))
        e1 = F.relu(F.conv2d(s, w_expand1, b_expand1, stride=1, padding=0))
        e3 = F.relu(F.conv2d(s, w_expand3, b_expand3, stride=1, padding=1))
        return torch.cat([e1, e3], dim=1)

def get_inputs():
    x = torch.rand(128, 3, 256, 256)
    w_squeeze = torch.rand(6, 3, 1, 1)
    b_squeeze = torch.rand(6)
    w_expand1 = torch.rand(64, 6, 1, 1)
    b_expand1 = torch.rand(64)
    w_expand3 = torch.rand(64, 6, 3, 3)
    b_expand3 = torch.rand(64)
    return [x, w_squeeze, b_squeeze, w_expand1, b_expand1, w_expand3, b_expand3]

def get_init_inputs():
    return []
