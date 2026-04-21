# 测试不同中间精度的效果
# 需要修改算子代码以支持不同的累加器精度

# 测试1：全 FP16
result_all_fp16 = run_operator_all_fp16(input_data)

# 测试2：累加器 FP32
result_fp32_accum = run_operator_fp32_accum(input_data)

# 对比
print(f"All FP16: max error = {np.abs(result_all_fp16 - expected).max():.2e}")
print(f"FP32 Accum: max error = {np.abs(result_fp32_accum - expected).max():.2e}")
