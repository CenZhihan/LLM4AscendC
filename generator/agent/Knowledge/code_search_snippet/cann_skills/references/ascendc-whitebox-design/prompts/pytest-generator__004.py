def make_data(shape, dtype, data_range):
    if data_range == "zero":
        return torch.zeros(shape, dtype=dtype)
    elif data_range == "extreme":
        return torch.full(shape, 65504.0 if dtype == torch.float16 else 3.4e38, dtype=dtype)
    elif data_range == "negative":
        return -torch.rand(shape, dtype=dtype) * 10
    elif data_range == "tiny_pos":
        return torch.ones(shape, dtype=dtype) * 1e-6
    elif data_range == "all_ones":
        return torch.ones(shape, dtype=dtype)
    elif data_range == "near_zero":
        return (torch.rand(shape, dtype=dtype) - 0.5) * 0.02
    elif data_range == "with_inf":
        t = torch.randn(shape, dtype=dtype)
        t.view(-1)[0] = float('inf')
        return t
    elif data_range == "with_nan":
        t = torch.randn(shape, dtype=dtype)
        t.view(-1)[0] = float('nan')
        return t
    else:  # normal
        return torch.randn(shape, dtype=dtype)
