// ✅ 正确：完整的 API 注释
// API: ReduceMax
// 功能: 求当前行的最大值
// 参数:
//   - dst: scalarLocal - 输出最大值（单个标量）
//   - src: xLocal[rowOffset] - 输入数据（当前行）
//   - tmp: reduceTmp - 临时计算 buffer
//   - count: rLength - 有效数据个数（非对齐长度）
//   - calIndex: false - 不计算索引
AscendC::ReduceMax<T>(scalarLocal, xLocal[rowOffset], reduceTmp, 
    static_cast<int32_t>(rLength), false);

// ❌ 错误：缺少注释
AscendC::ReduceMax<T>(scalarLocal, xLocal[rowOffset], reduceTmp, 
    static_cast<int32_t>(rLength), false);  // ⛔️ 无参数说明

// ❌ 错误：注释不完整
// 求 max
AscendC::ReduceMax<T>(scalarLocal, xLocal[rowOffset], reduceTmp, 
    static_cast<int32_t>(rLength), false);  // ⛔️ 缺少参数说明
