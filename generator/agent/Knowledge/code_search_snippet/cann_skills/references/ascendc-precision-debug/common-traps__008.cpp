// 使用 FP32 累加器
float sum_fp32 = 0.0f;
for (int i = 0; i < size; ++i) {
    sum_fp32 += static_cast<float>(input[i]);
}
output = static_cast<half>(sum_fp32);

// ReduceMax/ReduceMin 不受影响，可以保持 FP16
half max_val = ReduceMax(input);  // FP16 足够

// ReduceSum/ReduceMean 建议使用 FP32
float mean_fp32 = 0.0f;
for (int i = 0; i < size; ++i) {
    mean_fp32 += static_cast<float>(input[i]);
}
mean_fp32 /= static_cast<float>(size);
