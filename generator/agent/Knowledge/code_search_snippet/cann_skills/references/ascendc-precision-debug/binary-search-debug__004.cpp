// 第1步：打印输入（前N个元素）
printf("Input samples: ");
for (int i = 0; i < min(5, size); ++i) {
    printf("%.6f ", static_cast<float>(input[i]));
}
printf("...\n");

// 第2步：验证累加过程（分段打印）
float sum_fp32 = 0.0f;
for (int i = 0; i < size; ++i) {
    float val = static_cast<float>(input[i]);
    sum_fp32 += val;

    // 每100个元素打印一次
    if ((i + 1) % 100 == 0 || i == size - 1) {
        printf("Step2 - accumulated %d elements: sum = %.6f\n",
               i + 1, sum_fp32);
    }
}

// 第3步：验证最终输出
half output = static_cast<half>(sum_fp32);
printf("Step3 - final output = %.6f\n", static_cast<float>(output));
