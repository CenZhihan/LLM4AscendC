# 从 FP32 开始，逐步降精度到 FP16
input_data = np.random.rand(8, 16).astype(np.float64)  # 最高精度
expected = softmax_cpu(input_data)

for dtype in [np.float32, np.float16]:
    result = run_operator(input_data.astype(dtype))
    error = np.abs(result - expected)

    print(f"dtype={dtype.__name__}:")
    print(f"  Max error: {error.max():.2e}")
    print(f"  Mean error: {error.mean():.2e}")
