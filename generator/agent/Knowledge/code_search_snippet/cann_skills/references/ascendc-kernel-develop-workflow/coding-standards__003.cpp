// Host 侧 - 纯向量算子示例
int64_t availableCoreNum = 8;  // 默认值
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
uint32_t usedNumBlocks = (totalRows < availableCoreNum) ? totalRows : (uint32_t)availableCoreNum;

// Host 侧 - 矩阵算子示例
int64_t availableCoreNum = 8;
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &availableCoreNum);

// Tiling 结构体
struct CustomTiling {
    uint32_t totalRows;       // 总行数
    uint32_t rowsPerCore;     // 每核处理行数
    uint32_t rowsTail;        // 尾核行数
    uint32_t usedCoreNum;     // 实际使用核数
};

// Kernel 侧
blockIdx = AscendC::GetBlockIdx();
if (blockIdx >= tiling.usedCoreNum) return;  // 越界检查
