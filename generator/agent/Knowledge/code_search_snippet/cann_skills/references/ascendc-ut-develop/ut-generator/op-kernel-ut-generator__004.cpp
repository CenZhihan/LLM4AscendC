#include "tiling_case_executor.h"

TEST_F(IsInfTest, test_case_float16_1) {
    // 1. 构造Tiling上下文并获取参数
    optiling::IsInfCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("IsInf",
        {{{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{128, 64}, {128, 64}}, ge::DT_BOOL, ge::FORMAT_ND}},
        &compileInfo);

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);

    // 2. 使用TilingInfo中的参数
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    // 3. 执行
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    ICPU_RUN_KF(is_inf, tilingInfo.blockNum, x, y, workspace, tiling);
}
