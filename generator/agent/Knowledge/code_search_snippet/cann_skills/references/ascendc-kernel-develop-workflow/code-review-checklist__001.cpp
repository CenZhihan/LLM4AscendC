// ✅ 正确（Host 侧计算)
// common.h
inline void computeSoftmaxTiling(sftmaxTilingData& tiling, ...) {
    tiling.rowsPerLoop = ...;  // 在 Host 侧计算
}

// xxx.asc
int32_t main() {
    sftmaxTilingData tiling;
    computeSoftmaxTiling(tiling, 4, 128);  // Host 侧计算
    KernelCall(..., (uint8_t*)&tiling);
}

// xxx_ar_fullload.h
__aicore__ inline void Init(..., const sftmaxTilingData& tiling) {
    rowsPerLoop = tiling.rowsPerLoop;  // Kernel 直接使用
}

// ❌ 错误（Kernel 中计算)
__aicore__ inline void Init(...) {
    uint32_t ubPerRow = rLengthAlign * sizeof(T);
    rowsPerLoop = availableUB / (2 * ubPerRow);  // ⛔️ 浪费计算资源
}
