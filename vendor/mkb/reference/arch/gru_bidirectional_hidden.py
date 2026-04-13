import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(128, 256, 6, batch_first=False, bidirectional=True)

    def forward(self, x, h0, w_ih, w_hh, b_ih, b_hh, ybuf):
        _, hn = self.rnn(x.float(), h0.float())
        return hn.half()

T, B, I, H, L = 512, 10, 128, 256, 6
IN2H = 2 * H

def get_inputs():
    x = torch.rand(T, B, I, dtype=torch.float16).contiguous()
    h0 = torch.rand(L * 2, B, H, dtype=torch.float16).contiguous()
    w_ih = torch.rand(L * 2 * 3 * H, IN2H, dtype=torch.float16).contiguous()
    w_hh = torch.rand(L * 2 * 3 * H, H, dtype=torch.float16).contiguous()
    b_ih = torch.rand(L * 2 * 3 * H, dtype=torch.float16).contiguous()
    b_hh = torch.rand(L * 2 * 3 * H, dtype=torch.float16).contiguous()
    ybuf = torch.rand(L, T, B, IN2H, dtype=torch.float16).contiguous()
    return [x, h0, w_ih, w_hh, b_ih, b_hh, ybuf]

def get_init_inputs():
    return []
