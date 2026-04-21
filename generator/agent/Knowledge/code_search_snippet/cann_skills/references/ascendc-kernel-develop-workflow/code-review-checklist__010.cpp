// ✅ 正确（使用 DataCopyPad）
// CopyIn: GM → UB
AscendC::DataCopyPad(xLocal, xGm[tiling.startIdx], 
    {rowsThisLoop, tiling.rLength, tiling.rLengthAlign, 0, 0});

// CopyOut: UB → GM
AscendC::DataCopyPad(yGm[tiling.startIdx], yLocal,
    {rowsThisLoop, tiling.rLength, tiling.rLengthAlign, 0, 0});

// ❌ 错误（使用 DataCopy，无法处理非对齐）⛔️ 违反即不通过
AscendC::DataCopy(xLocal, xGm[...], rowsThisLoop * tiling.rLengthAlign);  // ⛔️ 审查不通过

// ❌ 错误（使用 GlobalTensor::SetValue/GetValue，效率极低）
xGm.SetValue(idx, value);   // ⛔️ 仅用于调试，生产代码禁止使用
T val = xGm.GetValue(idx);  // ⛔️ 仅用于调试，生产代码禁止使用

// ❌ 错误（手动循环，效率低）
for (uint32_t i = 0; i < rowsThisLoop; i++) {
    for (uint32_t j = 0; j < tiling.rLength; j++) {
        xLocal.SetValue(i * tiling.rLengthAlign + j, xGm.GetValue(...));  // ⛔️
    }
}
