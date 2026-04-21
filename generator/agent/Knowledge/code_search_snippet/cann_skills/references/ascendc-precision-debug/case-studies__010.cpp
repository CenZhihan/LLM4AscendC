half SinhStable(half x) {
    // 使用 FP32 进行中间计算
    float x_f32 = static_cast<float>(x);

    float exp_x = exp(x_f32);
    float exp_neg_x = exp(-x_f32);
    float numerator_f32 = exp_x - exp_neg_x;

    return static_cast<half>(numerator_f32 / 2.0f);
}
