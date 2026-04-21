// 直接 exp 会溢出
half x = 100.0h;
half exp_x = Exp(x);
printf("exp(%.1f) = %f\n", static_cast<float>(x), static_cast<float>(exp_x));
// 输出: exp(100.0) = inf
