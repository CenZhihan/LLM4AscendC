// 完整二分拆解代码

// 第1步：验证 exp(x)
half exp_x = Exp(input);
printf("Step1 - exp(%.6f) = %.6f\n",
       static_cast<float>(input),
       static_cast<float>(exp_x));

// 第2步：验证 exp(-x)
half exp_neg_x = Exp(-input);
printf("Step2 - exp(%.6f) = %.6f\n",
       static_cast<float>(-input),
       static_cast<float>(exp_neg_x));

// 第3步：验证分子减法
half numerator = exp_x - exp_neg_x;
printf("Step3 - numerator = %.6f - %.6f = %.6f\n",
       static_cast<float>(exp_x),
       static_cast<float>(exp_neg_x),
       static_cast<float>(numerator));

// 第4步：验证最终除法
half result = numerator / 2.0h;
printf("Step4 - result = %.6f / 2 = %.6f\n",
       static_cast<float>(numerator),
       static_cast<float>(result));
