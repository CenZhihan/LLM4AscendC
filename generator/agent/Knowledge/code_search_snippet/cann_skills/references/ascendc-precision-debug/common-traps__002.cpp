// 先减去最大值，再 exp，避免溢出
half max_val = input[0];
for (int i = 1; i < size; ++i) {
    max_val = max(max_val, input[i]);
}

half exp_sum = 0.0h;
for (int i = 0; i < size; ++i) {
    half shifted = input[i] - max_val;  // 关键！使最大输入为0
    half exp_val = Exp(shifted);
    exp_sum += exp_val;
    output[i] = exp_val;
}

for (int i = 0; i < size; ++i) {
    output[i] = output[i] / exp_sum;
}
