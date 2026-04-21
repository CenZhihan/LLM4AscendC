void TraceSteps(const char* step, half value) {
    printf("[STEP] %s: value=%.6f\n", step, static_cast<float>(value));
}

// 使用
TraceSteps("initial", input);
TraceSteps("after_exp", exp_result);
TraceSteps("after_sum", sum_result);
TraceSteps("final", output);
