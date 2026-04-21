# 生成测试数据
size = 1000
input_data = np.ones(size, dtype=np.float16) * 0.1

# 期望结果
expected = 100.0  # 1000 * 0.1

# 实际结果
result = reducesum_fp16(input_data)
print(f"Expected: {expected}, Got: {result}, Error: {abs(result - expected)}")
