// 使用 FP32 进行减法运算
float diff_fp32 = static_cast<float>(a) - static_cast<float>(b);
half result = static_cast<half>(diff_fp32);
