def is_32byte_aligned(shape, dtype):
    """检查形状尾轴是否32字节对齐"""
    element_size = np.dtype(dtype).itemsize
    last_dim = shape[-1]
    return (last_dim * element_size) % 32 == 0

# 示例
print(is_32byte_aligned((8, 16), np.float16))  # True: 16*2=32字节
print(is_32byte_aligned((8, 8), np.float32))   # True: 8*4=32字节
print(is_32byte_aligned((8, 17), np.float16))  # False: 17*2=34字节
