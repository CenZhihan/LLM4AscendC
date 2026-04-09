import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, anchor, positive, negative, margin):
        m = float(margin.reshape(-1)[0].item()) if margin.numel() else 0.0
        return F.triplet_margin_loss(anchor, positive, negative, margin=m)

def get_inputs():
    n = 32
    d = 128
    anchor = torch.randn(n, d)
    positive = torch.randn(n, d)
    negative = torch.randn(n, d)
    margin = torch.tensor(1.0)
    return [anchor, positive, negative, margin]

def get_init_inputs():
    return []
