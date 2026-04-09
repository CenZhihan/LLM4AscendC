import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Deep narrow MLP with packed weights (pybind: x, w_packed, b_packed).
    """
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_layer_sizes[0]
        self.num_hidden = len(hidden_layer_sizes)
        self.output_size = output_size

    def forward(self, x, w_packed, b_packed):
        in_sz = self.input_size
        hs = self.hidden_size
        nh = self.num_hidden
        out_sz = self.output_size
        p = 0
        pb = 0
        w0n = hs * in_sz
        w0 = w_packed[p : p + w0n].view(hs, in_sz)
        p += w0n
        b0 = b_packed[pb : pb + hs]
        pb += hs
        h = F.linear(x, w0, b0)
        h = F.relu(h)
        for _ in range(nh - 1):
            w_blk = w_packed[p : p + hs * hs].view(hs, hs)
            p += hs * hs
            bi = b_packed[pb : pb + hs]
            pb += hs
            h = F.linear(h, w_blk, bi)
            h = F.relu(h)
        w_last = w_packed[p : p + out_sz * hs].view(out_sz, hs)
        p += out_sz * hs
        b_last = b_packed[pb : pb + out_sz]
        return F.linear(h, w_last, b_last)

batch_size = 1024
input_size = 8192
hidden_layer_sizes = [1024] * 16
output_size = 8192

def get_inputs():
    x = torch.rand(batch_size, input_size)
    hs = hidden_layer_sizes[0]
    nh = len(hidden_layer_sizes)
    out_sz = output_size
    in_sz = input_size
    chunks = []
    chunks.append(torch.rand(hs, in_sz))
    for _ in range(nh - 1):
        chunks.append(torch.rand(hs, hs))
    chunks.append(torch.rand(out_sz, hs))
    w_packed = torch.cat([c.flatten() for c in chunks])
    b_chunks = [torch.rand(hs) for _ in range(nh)]
    b_chunks.append(torch.rand(out_sz))
    b_packed = torch.cat(b_chunks)
    return [x, w_packed, b_packed]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]
