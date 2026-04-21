// ✅ 正确：half → float 使用 CAST_NONE
AscendC::Cast<float, half>(xLocal, xLocalHalf, AscendC::RoundMode::CAST_NONE, cols);

// ✅ 正确：float → half 使用 CAST_ROUND
AscendC::Cast<half, float>(yLocalHalf, xLocal, AscendC::RoundMode::CAST_ROUND, cols);
