half SoftmaxStable(half* input, int size) {
    // 先求最大值
    half max_val = input[0];
    for (int i = 1; i < size; ++i) {
        max_val = max(max_val, input[i]);
    }

    // 计算 exp(x - max)，避免溢出
    float exp_sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        half shifted = input[i] - max_val;  // 最大输入变为 0
        exp_sum += static_cast<float>(Exp(shifted));
    }

    // 归一化
    for (int i = 0; i < size; ++i) {
        half shifted = input[i] - max_val;
        output[i] = Exp(shifted) / static_cast<half>(exp_sum);
    }
}
