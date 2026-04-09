import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x,h0,c0,w_ih,w_hh,b_ih,b_hh,fc_w,fc_b) -> [B,O]. Weights follow kernel packing; reference uses nn.LSTM+Linear of same dims."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256, 6, batch_first=True)
        self.fc = nn.Linear(256, 10)

    def forward(self, x, h0, c0, w_ih, w_hh, b_ih, b_hh, fc_w, fc_b):
        out, _ = self.lstm(x, (h0, c0))
        return F.linear(out[:, -1, :], fc_w, fc_b)

B, S, I, H, L, O = 10, 512, 128, 256, 6, 10
ROWS = L * 4 * H

def get_inputs():
    x = torch.rand(B, S, I)
    h0 = torch.rand(L, B, H)
    c0 = torch.rand(L, B, H)
    w_ih = torch.rand(ROWS, H)
    w_hh = torch.rand(ROWS, H)
    b_ih = torch.rand(ROWS)
    b_hh = torch.rand(ROWS)
    fc_w = torch.rand(O, H)
    fc_b = torch.rand(O)
    return [x, h0, c0, w_ih, w_hh, b_ih, b_hh, fc_w, fc_b]

def get_init_inputs():
    return []
