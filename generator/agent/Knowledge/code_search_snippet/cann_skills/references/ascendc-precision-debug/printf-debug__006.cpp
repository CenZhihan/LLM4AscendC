// 并排打印期望值和实际值
for (int i = 0; i < size; i += 10) {
    printf("[%d] got=%.6f, exp=%.6f, diff=%.2e\n",
           i,
           static_cast<float>(output[i]),
           static_cast<float>(expected[i]),
           static_cast<float>(abs(output[i] - expected[i])));
}
