// 第0步：打印输入（前3个元素）
printf("Input samples: [%.6f, %.6f, %.6f]\n",
       static_cast<float>(input[0]),
       static_cast<float>(input[1]),
       static_cast<float>(input[2]));

// 第1步：验证 ReduceMax
half max_val = ReduceMax(input);
printf("Step1 - max_val = %.6f\n", static_cast<float>(max_val));

// 第2步：验证广播后的 Sub（前3个元素）
for (int i = 0; i < 3 && i < size; ++i) {
    half shifted = input[i] - max_val;
    printf("Step2 - shifted[%d] = %.6f - %.6f = %.6f\n",
           i,
           static_cast<float>(input[i]),
           static_cast<float>(max_val),
           static_cast<float>(shifted));
}

// 第3步：验证 Exp（前3个元素）
LocalTensor<half> exp_vals;
// ... 分配内存 ...
for (int i = 0; i < 3 && i < size; ++i) {
    half shifted = input[i] - max_val;
    half exp_val = Exp(shifted);
    exp_vals[i] = exp_val;
    printf("Step3 - exp(%.6f) = %.6f\n",
           static_cast<float>(shifted),
           static_cast<float>(exp_val));
}

// 第4步：验证 ReduceSum
half exp_sum = ReduceSum(exp_vals);
printf("Step4 - exp_sum = %.6f\n", static_cast<float>(exp_sum));

// 第5步：验证归一化（前3个元素）
for (int i = 0; i < 3 && i < size; ++i) {
    half output_val = exp_vals[i] / exp_sum;
    output[i] = output_val;
    printf("Step5 - output[%d] = %.6f / %.6f = %.6f\n",
           i,
           static_cast<float>(exp_vals[i]),
           static_cast<float>(exp_sum),
           static_cast<float>(output_val));
}

// 验证：输出之和应该接近1
half output_sum = ReduceSum(output);
printf("Verification - output_sum = %.6f (expected: 1.0)\n",
       static_cast<float>(output_sum));
