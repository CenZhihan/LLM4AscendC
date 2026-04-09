import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(128, 256, 6, batch_first=False)

    def forward(self, x, h0, w_ih, w_hh, b_ih, b_hh):
        y, _ = self.rnn(x, h0)
        return y

T, B, I, H, L = 512, 10, 128, 256, 6

def get_inputs():
    x = torch.rand(T, B, I)
    h0 = torch.rand(L, B, H)
    w_ih = torch.rand(L * 3 * H, H)
    w_hh = torch.rand(L * 3 * H, H)
    b_ih = torch.rand(L * 3 * H)
    b_hh = torch.rand(L * 3 * H)
    return [x, h0, w_ih, w_hh, b_ih, b_hh]

def get_init_inputs():
    return []
