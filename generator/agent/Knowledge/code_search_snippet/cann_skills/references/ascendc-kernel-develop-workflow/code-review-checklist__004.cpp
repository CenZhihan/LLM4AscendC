// ✅ 正确：动态计算核数，避免浪费 ⚠️ 重要：必须在 aclrtSetDevice 接口后调用
// Host 侧 - 纯向量算子示例
uint32_t totalRows = shape[0];
int64_t availableCoreNum = 8;  // 默认值
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);

// 关键：数据量与核数匹配
uint32_t usedNumBlocks = (totalRows < availableCoreNum) ? totalRows : (uint32_t)availableCoreNum;

// 特殊情况：数据量极小时使用单核
if (totalRows <= 8) {
    usedNumBlocks = 1;  // 8个元素用1核处理，避免7核浪费
}

tiling.usedCoreNum = usedNumBlocks;
tiling.rowsPerCore = totalRows / usedNumBlocks;
tiling.rowsTail = totalRows % usedNumBlocks;

// Kernel 侧
blockIdx = AscendC::GetBlockIdx();
if (blockIdx >= tiling.usedCoreNum) {
    return;  // 越界检查
}

// 每个核处理不同数据段（startIdx 不同）
uint32_t startIdx = blockIdx * tiling.rowsPerCore;
uint32_t rowsThisCore = (blockIdx == tiling.usedCoreNum - 1 && tiling.rowsTail != 0) 
                        ? tiling.rowsTail : tiling.rowsPerCore;
