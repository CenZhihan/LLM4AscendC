// 方法1：添加小常数（Epsilon）
half eps = 1e-7h;
half safe_div = numerator / (denominator + eps);

// 方法2：条件判断
half eps = 1e-7h;
half safe_div = (abs(denominator) < eps) ? 0.0h : numerator / denominator;

// 方法3：使用最大值保护
half safe_div = numerator / max(denominator, eps);
