// 只打印前N个元素
const int PRINT_N = 3;
for (int i = 0; i < PRINT_N && i < size; ++i) {
    printf("arr[%d] = %.6f\n", i, static_cast<float>(arr[i]));
}

// 条件打印：只打印大误差位置
for (int i = 0; i < size; ++i) {
    if (abs(output[i] - expected[i]) > threshold) {
        printf("Mismatch @%d: got %.6f, exp %.6f\n",
               i, output[i], expected[i]);
    }
}

// 采样打印：每隔N个打印一个
for (int i = 0; i < size; i += 100) {
    printf("arr[%d] = %.6f\n", i, static_cast<float>(arr[i]));
}
