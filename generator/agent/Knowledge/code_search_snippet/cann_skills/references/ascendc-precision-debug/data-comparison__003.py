# 非对齐输入，测试是否需要特殊处理
test_input_unaligned = np.random.rand(8, 8, 9).astype(np.float32)  # 尾轴9，非32字节对齐

# 或
test_input_unaligned = np.random.rand(8, 8, 17).astype(np.float32)  # 尾轴17
