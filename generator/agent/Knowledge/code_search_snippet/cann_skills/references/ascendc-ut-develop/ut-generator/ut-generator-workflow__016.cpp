#include "tikicpulib.h"

extern "C" __global__ __aicore__ void <op>(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

// 分配内存
uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(<Op>TilingData));

// 设置 Tiling 数据
<Op>TilingData* tilingData = reinterpret_cast<<Op>TilingData*>(tiling);
tilingData->usedCoreNum = 1;

// 执行 Kernel
ICPU_SET_TILING_KEY(tilingKey);
ICPU_RUN_KF(<op>, numBlocks, x, y, workspace, tilingData);

// 释放内存
AscendC::GmFree((void*)x);
