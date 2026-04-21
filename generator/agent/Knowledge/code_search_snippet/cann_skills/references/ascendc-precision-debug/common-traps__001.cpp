// 关键中间值用 FP32
float sum_fp32 = 0.0f;
for (int i = 0; i < n; ++i) {
    sum_fp32 += static_cast<float>(values[i]);  // 先转为 FP32 累加
}
output = static_cast<half>(sum_fp32);  // 最后转回 FP16
