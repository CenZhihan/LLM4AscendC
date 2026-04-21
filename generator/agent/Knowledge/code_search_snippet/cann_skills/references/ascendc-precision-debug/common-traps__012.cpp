// ❌ 错误：half → float 使用 CAST_ROUND
AscendC::Cast<float, half>(xLocal, xLocalHalf, AscendC::RoundMode::CAST_ROUND, cols);
// 结果：数据完全错误，多行输出相同
