void LocateErrors(half* output, half* expected, int size) {
    int error_count = 0;
    half max_error = 0.0h;
    int max_error_idx = -1;

    for (int i = 0; i < size; ++i) {
        half error = abs(output[i] - expected[i]);
        if (error > 1e-3h) {
            error_count++;
            if (error > max_error) {
                max_error = error;
                max_error_idx = i;
            }
        }
    }

    printf("[ERRORS] count=%d, max_error=%.2e @%d\n",
           error_count,
           static_cast<float>(max_error),
           max_error_idx);
}
