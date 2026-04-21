// 检查硬件约束
if (cols < 8) {
    printf("Error: cols must be >= 8 (got %d)\n", cols);
    return;
}

// 检查数值范围
for (int i = 0; i < size; ++i) {
    if (isinf(static_cast<float>(input[i]))) {
        printf("Warning: input[%d] is Inf\n", i);
    }
}
