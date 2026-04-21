// Softmax: 输出之和应该等于1
half output_sum = ReduceSum(output);
printf("Softmax check: sum(output) = %.6f (expected: 1.0)\n",
       static_cast<float>(output_sum));

// ReLU: 输出不应该有负值
bool has_negative = false;
for (int i = 0; i < size; ++i) {
    if (output[i] < 0.0h) {
        has_negative = true;
        break;
    }
}
printf("ReLU check: has_negative = %s\n", has_negative ? "true" : "false");

// 对称性：sin(-x) = -sin(x)
half sin_x = Sin(x);
half sin_neg_x = Sin(-x);
half symmetry_sum = sin_x + sin_neg_x;
printf("Symmetry check: sin(%.2f) + sin(%.2f) = %.6f (expected: 0)\n",
       static_cast<float>(x),
       static_cast<float>(-x),
       static_cast<float>(symmetry_sum));
