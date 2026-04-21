import numpy as np

# 32字节对齐 + FP32（排除对齐和精度问题）
# FP32: 32字节 = 8个元素
test_input_fp32 = np.random.rand(8, 8, 8).astype(np.float32)  # 尾轴8=8*4字节=32字节对齐

# 或使用简单值便于验证
test_input_fp32 = np.ones((8, 8, 8), dtype=np.float32)

# 或使用已知输入验证输出
test_input_fp32 = np.zeros((8, 8, 8), dtype=np.float32)
test_input_fp32[0, 0, 0] = 1.0  # 单个非零值
