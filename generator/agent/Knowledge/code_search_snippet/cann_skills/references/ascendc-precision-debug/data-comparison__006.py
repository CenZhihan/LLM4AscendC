def generate_aligned_test(shape, dtype):
    """生成32字节对齐的测试数据"""
    element_size = np.dtype(dtype).itemsize
    aligned_size = 32 // element_size

    # 调整尾轴为对齐大小的倍数
    adjusted_shape = list(shape)
    adjusted_shape[-1] = ((shape[-1] + aligned_size - 1) // aligned_size) * aligned_size

    return np.random.rand(*adjusted_shape).astype(dtype)

# 示例
test_data = generate_aligned_test((8, 10), np.float32)  # 尾轴调整为16（8的倍数）
print(f"Adjusted shape: {test_data.shape}")  # (8, 16)
