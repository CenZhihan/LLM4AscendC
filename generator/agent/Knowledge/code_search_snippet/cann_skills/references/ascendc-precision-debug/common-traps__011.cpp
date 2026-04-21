// 避免不必要的类型转换
// 不推荐：频繁转换
half temp = static_cast<half>(float_value);
float result = static_cast<float>(temp);

// 推荐：保持一种类型
float result = float_value;  // 尽量用 FP32 计算

// 只在必要时转换
half output = static_cast<half>(final_result_fp32);
