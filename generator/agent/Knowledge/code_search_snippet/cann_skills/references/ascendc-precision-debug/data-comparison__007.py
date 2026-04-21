boundary_cases = {
    "零值": 0.0,
    "极小值": 1e-10,
    "小值": 1e-6,
    "正常值": 1.0,
    "大值": 1e6,
    "极大值": 1e10,
    "负值": -1.0,
    "FP16饱和": 65504.0,     # FP16 最大值
    "FP16负饱和": -65504.0,  # FP16 最小值
}

# 生成边界测试数据
for name, value in boundary_cases.items():
    test_input = np.full((8, 8), value, dtype=np.float32)
    result = run_operator(test_input)
    print(f"{name}: input={value}, output={result[0, 0]}")
