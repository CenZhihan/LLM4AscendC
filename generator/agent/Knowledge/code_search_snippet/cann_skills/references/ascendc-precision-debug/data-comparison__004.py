# FP16 测试（保持对齐）
# FP16: 32字节 = 16个元素
test_input_fp16 = np.random.rand(8, 8, 16).astype(np.float16)  # 尾轴16=16*2字节=32字节对齐

# 对比 FP16 和 FP32 结果
result_fp32 = run_operator(test_input_fp32.astype(np.float32))
result_fp16 = run_operator(test_input_fp16.astype(np.float16))

# 分析精度差异
error = np.abs(result_fp32 - result_fp16)
print(f"FP16 vs FP32: max error = {error.max():.2e}")
