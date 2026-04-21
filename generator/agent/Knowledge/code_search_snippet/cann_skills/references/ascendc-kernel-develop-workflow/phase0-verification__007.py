import numpy as np

# 构造输入
x1 = np.random.randn(1024).astype(np.float32)
x2 = np.random.randn(1, 1, 2, 1).astype(np.float32)
x3 = np.random.randn(1).astype(np.float32)

# 按公式计算
output = x1 + value * (x2 / x3)

# 验证 shape
print(f"输出 shape: {output.shape}")
