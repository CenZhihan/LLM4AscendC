// 精度敏感：使用 FP32 避免累加误差
float sum_fp32 = 0.0f;
for (int i = 0; i < size; ++i) {
    sum_fp32 += static_cast<float>(input[i]);
}

// 精度敏感：先减 max 避免 exp 溢出
half max_val = ReduceMax(input);
half shifted = input[i] - max_val;
