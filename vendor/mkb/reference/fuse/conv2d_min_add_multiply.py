import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Pybind: (x, weight, conv_bias optional, post_bias [Cout,1,1], constant_value scalar,
      scaling_factor scalar).
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(Model, self).__init__()

    def forward(self, x, weight, conv_bias_opt, post_bias, constant_value, scaling_factor):
        cv = float(constant_value.reshape(-1)[0].item())
        sv = float(scaling_factor.reshape(-1)[0].item())
        conv_b = conv_bias_opt if conv_bias_opt is not None else None
        x = F.conv2d(x, weight, conv_b, stride=1, padding=0, dilation=1)
        cap = torch.as_tensor(cv, device=x.device, dtype=x.dtype)
        x = torch.minimum(x, cap)
        x = x + post_bias
        return x * sv

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    w = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    cb = torch.rand(out_channels)
    pb = torch.randn(bias_shape)
    cvt = torch.tensor(constant_value, dtype=torch.float32)
    svt = torch.tensor(scaling_factor, dtype=torch.float32)
    return [x, w, cb, pb, cvt, svt]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]
