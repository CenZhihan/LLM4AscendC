import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Pybind: matmul_dropout_softmax_custom(x, weight, bias, dropout_p, training).
    training: int32 scalar tensor (0/1).
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, weight, bias, dropout_p, training):
        p = float(dropout_p.detach().cpu().item())
        train = bool(int(training.detach().cpu().item()) != 0)
        x = F.linear(x, weight, bias)
        x = F.dropout(x, p=p, training=train)
        return F.softmax(x, dim=1)

batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2

def get_inputs():
    x = torch.rand(batch_size, in_features)
    w = torch.rand(out_features, in_features)
    b = torch.rand(out_features)
    dp = torch.tensor(dropout_p, dtype=torch.float32)
    tr = torch.tensor(1, dtype=torch.int32)
    return [x, w, b, dp, tr]

def get_init_inputs():
    return [in_features, out_features, dropout_p]
