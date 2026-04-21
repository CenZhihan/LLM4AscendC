TEST_F(add_lora_test, test_add_lora_0) {
    // 分配内存
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16781184);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AddLoraTilingData));

    // 手动设置Tiling参数
    AddLoraTilingData* tilingData = reinterpret_cast<AddLoraTilingData*>(tiling);
    tilingData->usedCoreNum = 20;
    tilingData->batch = 1;
    // ...

    // 执行
    ICPU_SET_TILING_KEY(100001);
    ICPU_RUN_KF(add_lora, 20, x, y, workspace, tilingData);

    // 释放
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}
