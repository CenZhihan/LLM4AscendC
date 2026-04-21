# FP32 测试
result_fp32 = sinh_fp32(test_input)
error_fp32 = np.abs(result_fp32 - expected)
print(f"FP32 max error: {error_fp32.max():.2e}")  # ~1e-6

# FP16 测试
result_fp16 = sinh_fp16(test_input)
error_fp16 = np.abs(result_fp16 - expected)
print(f"FP16 max error: {error_fp16.max():.2e}")  # ~1e-2
