#include "tikicpulib.h"

// 1. 声明kernel入口函数
extern "C" __global__ __aicore__ void <op>(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

// 2. 内存分配
uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(<Op>TilingData));

// 3. 设置Tiling数据
<Op>TilingData* tilingData = reinterpret_cast<<Op>TilingData*>(tiling);
tilingData->usedCoreNum = 1;
tilingData->totalLength = 1024;

// 4. 执行Kernel
ICPU_SET_TILING_KEY(tilingKey);
ICPU_RUN_KF(<op>, numBlocks, x, y, workspace, tilingData);

// 5. 释放内存
AscendC::GmFree((void*)x);
AscendC::GmFree((void*)y);
AscendC::GmFree((void*)workspace);
AscendC::GmFree((void*)tiling);
