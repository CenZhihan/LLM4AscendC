// 原始公式（数值不稳定）
half result = sqrt(x + 1) - sqrt(x);

// 稳定版本（有理化）
half result = 1.0h / (sqrt(x + 1) + sqrt(x));
