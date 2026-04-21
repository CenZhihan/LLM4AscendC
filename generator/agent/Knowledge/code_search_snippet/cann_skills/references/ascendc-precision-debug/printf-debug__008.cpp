// 打印最小值、最大值
half min_val = input[0];
half max_val = input[0];
float sum = 0.0f;

for (int i = 0; i < size; ++i) {
    min_val = min(min_val, input[i]);
    max_val = max(max_val, input[i]);
    sum += static_cast<float>(input[i]);
}

printf("Array stats: min=%.6f, max=%.6f, mean=%.6f\n",
       static_cast<float>(min_val),
       static_cast<float>(max_val),
       sum / static_cast<float>(size));

// 检查是否有 Inf/NaN
bool has_inf = false;
bool has_nan = false;
for (int i = 0; i < size; ++i) {
    float val = static_cast<float>(input[i]);
    if (isinf(val)) has_inf = true;
    if (isnan(val)) has_nan = true;
}
printf("Array checks: has_inf=%d, has_nan=%d\n", has_inf, has_nan);
