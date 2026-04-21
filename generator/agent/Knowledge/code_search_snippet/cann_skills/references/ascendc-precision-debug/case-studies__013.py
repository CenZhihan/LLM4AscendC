# 测试大输入
large_input = np.array([[100.0, 101.0, 102.0]], dtype=np.float16)
result = softmax(large_input)
print(result)  # [nan, nan, nan]
