half ReduceSumAccurate(half* input, int size) {
    // 使用 FP32 累加器
    float sum_fp32 = 0.0f;

    for (int i = 0; i < size; ++i) {
        sum_fp32 += static_cast<float>(input[i]);
    }

    return static_cast<half>(sum_fp32);
}
