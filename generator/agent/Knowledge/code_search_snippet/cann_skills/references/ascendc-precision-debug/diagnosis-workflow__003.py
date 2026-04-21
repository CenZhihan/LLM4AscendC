# 1. 单元素测试（最简单）
test_input = np.array([1.5], dtype=np.float32)

# 2. 小规模对齐测试（32字节对齐 + FP32）
test_input = np.random.rand(8, 16).astype(np.float32)  # 尾轴16=8*4字节

# 3. 边界值测试
boundary_cases = {
    "零值": 0.0,
    "极小值": 1e-10,
    "小值": 1e-6,
    "正常值": 1.0,
    "大值": 1e6,
    "极大值": 1e10,
    "负值": -1.0,
    "FP16饱和": 65504.0,
}

# 4. 非对齐测试
test_input = np.random.rand(8, 17).astype(np.float32)

# 5. FP16 精度测试
test_input = np.random.rand(8, 16).astype(np.float16)
