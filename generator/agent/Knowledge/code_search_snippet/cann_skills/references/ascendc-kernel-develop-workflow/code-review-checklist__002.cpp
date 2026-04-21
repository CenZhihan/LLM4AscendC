// ⚠️ 重要：必须先 aclrtSetDevice，再调用 aclrtGetDeviceInfo
aclError ret = aclrtSetDevice(deviceId);  // 先设置设备
if (ret != ACL_SUCCESS) {
    // 错误处理
}

// 获取设备核数（必须在 aclrtSetDevice 之后）
// 纯向量算子示例
int64_t availableCoreNum = 8;  // 默认值
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);

// 矩阵算子使用：ACL_DEV_ATTR_CUBE_CORE_NUM
// 混合算子使用：ACL_DEV_ATTR_AICORE_CORE_NUM

// 计算使用核数
uint32_t usedNumBlocks = (totalRows < availableCoreNum) ? totalRows : (uint32_t)availableCoreNum;

// Tiling 参数
tiling.usedCoreNum = usedNumBlocks;
tiling.rowsPerCore = totalRows / usedNumBlocks;
tiling.rowsTail = totalRows % usedNumBlocks;
