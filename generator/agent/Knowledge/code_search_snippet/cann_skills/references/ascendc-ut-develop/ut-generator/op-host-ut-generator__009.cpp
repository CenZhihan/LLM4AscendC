// 成功场景
TEST_F(AddTilingTest, test_tiling_fp16_001) {
    optiling::AddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Add",
        { {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND} },
        { {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND} },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 102, "...", {16777216});
}

// 失败场景
TEST_F(AddTilingTest, test_tiling_failed_dtype) {
    optiling::AddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Add",
        { {{{1, 64}, {1, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND} },  // 不支持的dtype
        { {{{1, 64}, {1, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND} },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}
