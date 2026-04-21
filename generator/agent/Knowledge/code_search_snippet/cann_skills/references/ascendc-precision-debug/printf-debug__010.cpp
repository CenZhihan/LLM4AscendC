// 在长循环中分段打印
for (int i = 0; i < size; ++i) {
    // ... 计算 ...

    // 每 1000 次迭代打印一次进度
    if ((i + 1) % 1000 == 0) {
        printf("Progress: %d/%d (%.1f%%)\n",
               i + 1, size, (i + 1) * 100.0f / size);
    }
}
