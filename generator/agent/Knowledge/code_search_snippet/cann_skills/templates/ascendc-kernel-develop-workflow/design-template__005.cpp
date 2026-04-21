// ❌ 错误: ARA 模板 (R×A0 数据， 归约 R 维
// 数据布局: [r0的全部A0, r1的全部A0, ..., r{R-1}的全部A0]
// 归约需求| 对每个 a0 位置，取 R 个值的 max
// 输出| A0 个最大值
 
// ⚠️ 错误: 使用 Level 2 API (无法处理 stride)
ReduceMax<T>(scalarLocal, xLocal[rowOffset], tmpLocal, rLength, false);
// 问题:
// - Level 2 API 只能处理连续的 count 个元素
// - 无法处理带 stride 的归约 (间隔 A0 个元素)
// - 导致: 数值错误、精度问题、崩溃
