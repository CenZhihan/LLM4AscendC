// 稳定的 log-sum-exp（用于 log-softmax）
half max_val = ReduceMax(input);
half sum_exp = 0.0h;
for (int i = 0; i < size; ++i) {
    sum_exp += Exp(input[i] - max_val);
}
output = max_val + Log(sum_exp);  // 数值稳定的 log(sum(exp(x)))

// 稳定的 sigmoid
half sigmoid = 1.0h / (1.0h + Exp(-x));

// 避免 exp(x) 过大的替代方案
// 如果知道 x 的范围，可以预先截断
half safe_exp = Exp(min(x, 10.0h));  // 限制最大指数为 10
