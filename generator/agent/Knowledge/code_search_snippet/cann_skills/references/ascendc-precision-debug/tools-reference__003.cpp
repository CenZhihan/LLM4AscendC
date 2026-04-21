#include "kernel_printf.h"

// 基础打印
printf("Value: %f\n", value);

// 指定小数位数
printf("Value: %.6f\n", value);     // 6位小数
printf("Value: %.2f\n", value);     // 2位小数

// 科学计数法
printf("Large: %.2e\n", large_value);

// 多个值
printf("x=%.6f, y=%.6f\n", x, y);

// 整数
printf("Index: %d\n", index);
printf("Size: %d x %d\n", height, width);

// 字符串
printf("Status: %s\n", "OK");

// 调试信息
printf("[DEBUG] Line %d: value=%.6f\n", __LINE__, value);

// FP16 需要转换
half h = 3.14h;
printf("Half: %.6f\n", static_cast<float>(h));
