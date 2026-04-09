import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Masked cumulative sum along one dimension.
    Pybind contract: masked_cumsum_custom(Tensor x, Tensor mask, int dim).
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, mask, dim: int):
        return torch.cumsum(x * mask, dim=dim)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    x = torch.rand(batch_size, *input_shape)
    mask = torch.randint(0, 2, x.shape).bool()
    return [x, mask, dim]

def get_init_inputs():
    return []
