// Host 侧（强制）- 纯向量算子示例
int64_t availableCoreNum = 8;  // 默认值
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
uint32_t usedNumBlocks = (totalRows < availableCoreNum) ? totalRows : (uint32_t)availableCoreNum;

// Kernel 侧（强制）
blockIdx = AscendC::GetBlockIdx();
if (blockIdx >= tiling.usedCoreNum) return;  // 越界检查
