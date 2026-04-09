import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256, 6, batch_first=True)

    def forward(self, x, h0, c0, w_ih, w_hh, b_ih, b_hh):
        _, (_, cn) = self.lstm(x, (h0, c0))
        return cn

B, S, I, H, L = 10, 512, 128, 256, 6
ROWS = L * 4 * H

def get_inputs():
    x = torch.rand(B, S, I)
    h0 = torch.rand(L, B, H)
    c0 = torch.rand(L, B, H)
    w_ih = torch.rand(ROWS, H)
    w_hh = torch.rand(ROWS, H)
    b_ih = torch.rand(ROWS)
    b_hh = torch.rand(ROWS)
    return [x, h0, c0, w_ih, w_hh, b_ih, b_hh]

def get_init_inputs():
    return []
