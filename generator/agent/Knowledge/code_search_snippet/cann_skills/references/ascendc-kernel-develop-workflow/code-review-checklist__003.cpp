// Kernel 侧越界检查
blockIdx = AscendC::GetBlockIdx();
if (blockIdx >= tiling.usedCoreNum) {
    return;  // ⚠️ 强制：超出使用核数直接返回
}

// 尾核处理
uint32_t rowsThisCore = (blockIdx == tiling.usedCoreNum - 1 && tiling.rowsTail != 0) 
                        ? tiling.rowsTail : tiling.rowsPerCore;
