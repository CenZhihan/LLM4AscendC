import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim):
        return x.sum(dim=int(dim))

batch_size = 128
features = 4096
length = 4095
reduce_dim = 1

def get_inputs():
    x = torch.rand(batch_size, features, length)
    return [x, reduce_dim]

def get_init_inputs():
    return []
