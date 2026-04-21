# 测试 Reduce 操作的最小元素数约束
reduce_sizes = [1, 2, 4, 8, 16, 32, 64]

for size in reduce_sizes:
    test_input = np.random.rand(8, size).astype(np.float32)
    result = run_operator(test_input)
    expected = reference_implementation(test_input)
    error = np.abs(result - expected)

    status = "PASS" if error.max() < 1e-5 else "FAIL"
    print(f"Reduce size {size}: {status}, max error = {error.max():.2e}")
