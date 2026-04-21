// 只打印前 N 个元素
const int PRINT_N = 3;
for (int i = 0; i < PRINT_N && i < size; ++i) {
    printf("arr[%d] = %.6f\n", i, static_cast<float>(arr[i]));
}

// 条件打印：只打印误差大的位置
half threshold = 1e-3h;
for (int i = 0; i < size; ++i) {
    if (abs(output[i] - expected[i]) > threshold) {
        printf("Mismatch @%d: got %.6f, exp %.6f, diff=%.2e\n",
               i,
               static_cast<float>(output[i]),
               static_cast<float>(expected[i]),
               static_cast<float>(abs(output[i] - expected[i])));
    }
}

// 采样打印：每隔 N 个打印一个
for (int i = 0; i < size; i += 100) {
    printf("arr[%d] = %.6f\n", i, static_cast<float>(arr[i]));
}
