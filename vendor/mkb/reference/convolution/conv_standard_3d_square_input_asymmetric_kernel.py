import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        # Custom op expects input layout [N,C,W,H,D]. PyTorch conv3d expects [N,C,D,H,W].
        # Convert layout before/after conv to align semantics.
        # x: [N,C,W,H,D]; weight from get_inputs: [Cout,Cin,Kw,Kh,Kd] (custom layout)
        x_torch = x.permute(0, 1, 4, 3, 2)  # -> [N,C,D,H,W]
        w_torch = weight.permute(0, 1, 4, 3, 2)  # [Cout,Cin,Kd,Kh,Kw]
        y_torch = F.conv3d(
            x_torch, w_torch, None,
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
            groups=1,
        )
        return y_torch.permute(0, 1, 4, 3, 2)

def get_inputs():
    x = torch.rand(16, 3, 64, 64, 64)
    w = torch.rand(64, 3, 3, 5, 7)
    return [x, w]

def get_init_inputs():
    return []
