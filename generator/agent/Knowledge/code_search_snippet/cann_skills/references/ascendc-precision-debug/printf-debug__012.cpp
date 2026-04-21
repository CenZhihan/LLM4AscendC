// 打印并验证条件
bool condition = /* ... */;
printf("[ASSERT] condition=%s (expected: true)\n",
       condition ? "true" : "false");

// 打印并验证值
half expected = 1.0h;
half actual = /* ... */;
printf("[VERIFY] expected=%.6f, actual=%.6f, match=%s\n",
       static_cast<float>(expected),
       static_cast<float>(actual),
       (abs(actual - expected) < 1e-6h) ? "true" : "false");
