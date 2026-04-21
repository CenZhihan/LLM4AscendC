// Softmax 公式: softmax(x) = exp(x-max) / sum(exp(x-max))

// 第1步：验证 ReduceMax
half max_val = ReduceMax(input);
printf("Step1 - max: %.6f\n", static_cast<float>(max_val));

// 第2步：验证广播后的 Sub
for (int i = 0; i < size; ++i) {
    half shifted = input[i] - max_val;
    if (i < 3) {
        printf("Step2 - shifted[%d]: %.6f (input=%.6f, max=%.6f)\n",
               i, static_cast<float>(shifted),
               static_cast<float>(input[i]), static_cast<float>(max_val));
    }
}

// 第3步：验证 Exp
for (int i = 0; i < size; ++i) {
    half exp_val = Exp(input[i] - max_val);
    if (i < 3) {
        printf("Step3 - exp[%d]: %.6f\n", i, static_cast<float>(exp_val));
    }
}

// 第4步：验证 ReduceSum
half exp_sum = ReduceSum(exp_values);
printf("Step4 - exp_sum: %.6f\n", static_cast<float>(exp_sum));

// 第5步：验证归一化
for (int i = 0; i < size; ++i) {
    output[i] = exp_values[i] / exp_sum;
    if (i < 3) {
        printf("Step5 - output[%d]: %.6f\n", i, static_cast<float>(output[i]));
    }
}

// 验证：输出之和应该接近1
half output_sum = ReduceSum(output);
printf("Verification - output_sum: %.6f (should be 1.0)\n",
       static_cast<float>(output_sum));
