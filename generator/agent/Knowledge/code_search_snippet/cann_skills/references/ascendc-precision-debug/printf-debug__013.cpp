void CheckValueRange(const char* name, half* arr, int size) {
    half min_val = arr[0];
    half max_val = arr[0];
    float sum = 0.0f;
    int inf_count = 0;
    int nan_count = 0;

    for (int i = 0; i < size; ++i) {
        min_val = min(min_val, arr[i]);
        max_val = max(max_val, arr[i]);
        sum += static_cast<float>(arr[i]);

        float val = static_cast<float>(arr[i]);
        if (isinf(val)) inf_count++;
        if (isnan(val)) nan_count++;
    }

    printf("[%s] min=%.6f, max=%.6f, mean=%.6f, inf=%d, nan=%d\n",
           name,
           static_cast<float>(min_val),
           static_cast<float>(max_val),
           sum / static_cast<float>(size),
           inf_count,
           nan_count);
}
