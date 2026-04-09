import torch
import torch.nn as nn

class Model(nn.Module):
    """Pybind: (x, h0, w_i2h, b_i2h, w_h2o, b_h2o) -> y [T,B,O]. One tanh cell + output projection."""

    def __init__(self):
        super().__init__()

    def forward(self, x, h0, w_i2h, b_i2h, w_h2o, b_h2o):
        T, B, I = x.shape
        H, K = w_i2h.shape
        h = h0
        outs = []
        for t in range(T):
            inp = torch.cat([x[t], h], dim=-1)
            pre = inp @ w_i2h.t() + b_i2h
            h = torch.tanh(pre)
            y_t = h @ w_h2o.t() + b_h2o
            outs.append(y_t)
        return torch.stack(outs, dim=0)

T, B, I = 256, 8, 1024
H, O = 256, 128
K = I + H

def get_inputs():
    x = torch.rand(T, B, I)
    h0 = torch.rand(B, H)
    w_i2h = torch.rand(H, K)
    b_i2h = torch.rand(H)
    w_h2o = torch.rand(O, H)
    b_h2o = torch.rand(O)
    return [x, h0, w_i2h, b_i2h, w_h2o, b_h2o]

def get_init_inputs():
    return []
