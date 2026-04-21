__aicore__ inline void ComputeFp16()
{
    // Step 1: half → float（低→高精度）
    AscendC::Cast<float, half>(xLocal, xLocalHalf, AscendC::RoundMode::CAST_NONE, cols);
    
    // Step 2: 在 FP32 上进行 softmax 计算
    // ... ReduceMax, Adds, Exp, ReduceSum, Muls ...
    
    // Step 3: float → half（高→低精度）
    AscendC::Cast<half, float>(yLocalHalf, xLocal, AscendC::RoundMode::CAST_ROUND, cols);
}
