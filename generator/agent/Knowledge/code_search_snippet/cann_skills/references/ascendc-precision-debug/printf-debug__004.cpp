half value = 3.14h;

// 错误：直接打印可能不准确
printf("Value: %f\n", value);

// 正确：先转换为 float
printf("Value: %.6f\n", static_cast<float>(value));
