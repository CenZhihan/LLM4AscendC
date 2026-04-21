// Kernel 内部获取启动的块数（官方推荐）
uint32_t blockDim = AscendC::GetBlockNum();

// Host 侧获取核数（用于 tiling 配置）
auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
uint32_t blockDim = platform->GetCoreNum();

// TILE_LENGTH 基于实际 UB 容量计算
uint32_t tileLength = CalculateTileSize(ubSize);  // ✅
