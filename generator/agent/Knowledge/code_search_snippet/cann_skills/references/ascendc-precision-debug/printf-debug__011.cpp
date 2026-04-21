half Compute(half x) {
    printf("[ENTER] Compute(%.6f)\n", static_cast<float>(x));

    // ... 计算 ...

    printf("[EXIT] Compute() -> %.6f\n", static_cast<float>(result));
    return result;
}
