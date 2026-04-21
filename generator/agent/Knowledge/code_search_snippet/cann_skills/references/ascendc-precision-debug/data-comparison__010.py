# CPU 参考（使用 NumPy）
def softmax_cpu(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# NPU 结果（从算子获取）
npu_output = run_operator_on_npu(input_data)

# 对比
cpu_output = softmax_cpu(input_data)
error = np.abs(npu_output - cpu_output)

print(f"Max error: {error.max():.2e}")
print(f"Mean error: {error.mean():.2e}")

# 找出最大误差位置
max_error_idx = error.argmax()
print(f"Worst case @ {max_error_idx}:")
print(f"  CPU: {cpu_output.flatten()[max_error_idx]}")
print(f"  NPU: {npu_output.flatten()[max_error_idx]}")
