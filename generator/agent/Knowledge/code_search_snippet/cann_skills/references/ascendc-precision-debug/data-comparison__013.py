# 测试不同规模下的精度
test_shapes = [
    (8, 8),      # 小规模
    (16, 16),    # 中小规模
    (32, 32),    # 中等规模
    (64, 64),    # 中大规模
    (128, 128),  # 大规模
    (256, 256),  # 超大规模
]

for shape in test_shapes:
    test_input = np.random.rand(*shape).astype(np.float32)
    result = run_operator(test_input)
    expected = reference_implementation(test_input)
    error = np.abs(result - expected)

    print(f"Shape {shape}: max error = {error.max():.2e}")
