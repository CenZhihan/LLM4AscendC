// 拆解计算步骤
half x = 1.5h;

// 第1步：exp(x)
half exp_x = Exp(x);
printf("exp(%.2f) = %.6f\n", static_cast<float>(x), static_cast<float>(exp_x));

// 第2步：exp(-x)
half exp_neg_x = Exp(-x);
printf("exp(%.2f) = %.6f\n", static_cast<float>(-x), static_cast<float>(exp_neg_x));

// 第3步：减法
half numerator = exp_x - exp_neg_x;
printf("numerator = %.6f - %.6f = %.6f\n",
       static_cast<float>(exp_x),
       static_cast<float>(exp_neg_x),
       static_cast<float>(numerator));

// 第4步：除法
half result = numerator / 2.0h;
printf("result = %.6f / 2 = %.6f\n",
       static_cast<float>(numerator),
       static_cast<float>(result));
