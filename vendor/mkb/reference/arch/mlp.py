import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    MLP with packed weights (matches pybind: x, w_packed, b_packed).
    """
    def __init__(self, input_size, layer_sizes, output_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.layer_sizes = list(layer_sizes)
        self.output_size = output_size

    def forward(self, x, w_packed, b_packed):
        h1, h2 = self.layer_sizes[0], self.layer_sizes[1]
        in_sz = self.input_size
        out_sz = self.output_size
        p = 0
        w1n = h1 * in_sz
        w1 = w_packed[p : p + w1n].view(h1, in_sz)
        p += w1n
        w2n = h2 * h1
        w2 = w_packed[p : p + w2n].view(h2, h1)
        p += w2n
        w3n = out_sz * h2
        w3 = w_packed[p : p + w3n].view(out_sz, h2)
        p += w3n
        pb = 0
        b1 = b_packed[pb : pb + h1]
        pb += h1
        b2 = b_packed[pb : pb + h2]
        pb += h2
        b3 = b_packed[pb : pb + out_sz]
        x = F.linear(x, w1, b1)
        x = F.relu(x)
        x = F.linear(x, w2, b2)
        x = F.relu(x)
        x = F.linear(x, w3, b3)
        return x

batch_size = 128
input_size = 16384
layer_sizes = [16384, 16384]
output_size = 8192

def get_inputs():
    x = torch.rand(batch_size, input_size)
    h1, h2 = layer_sizes
    w1 = torch.rand(h1, input_size)
    w2 = torch.rand(h2, h1)
    w3 = torch.rand(output_size, h2)
    b1 = torch.rand(h1)
    b2 = torch.rand(h2)
    b3 = torch.rand(output_size)
    w_packed = torch.cat([w1.flatten(), w2.flatten(), w3.flatten()])
    b_packed = torch.cat([b1, b2, b3])
    return [x, w_packed, b_packed]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]
