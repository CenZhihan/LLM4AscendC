// ✅ 正确: ARA 模板 (R×A0 数据)
// 数据布局| [r0的全部A0, r1的全部A0, ..., r{R-1}的全部A0]
// 归约需求| 对每个 a0 位置,取 R 个值的 max
// 输出| A0 个最大值
 
// ✅ 使用 Pattern API (可以处理 stride)
uint32_t alignedA0 = ((a0Count * sizeof(T) + 31) / 32) * 32 / sizeof(T);
uint32_t srcShape[] = {rLength, alignedA0};  // ⚠️ srcShape[1] 必须 32 字节对齐
AscendC::LocalTensor<T> maxLocal = maxSumBuf.Get<T>();
AscendC::LocalTensor<T> tmpLocal = tmpBuf.Get<T>();
 
// ✅ Pattern::RA - 沿第一维(R 维)归约
AscendC::ReduceMax<T, AscendC::Pattern::Reduce::RA>(
    maxLocal,           // 输出: A0 个最大值
    xLocal,             // 输入: R×A0 数据
    tmpLocal,           // 临时 buffer
    srcShape,            // {rLength, alignedA0}
    true                // srcInnerPad = true (有填充)
);
// 输出: A0 个最大值
