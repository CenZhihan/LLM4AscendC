#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

// 1. CompileInfo（必须先检测类型）
optiling::AddCompileInfo compileInfo = {64, 245760};
// 或
Ops::Base::BroadcastCompileInfo compileInfo = {64, 245760};

// 2. TilingContextPara
gert::TilingContextPara tilingContextPara("Add",
    { {{{1, 64}, {1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND} },  // 输入
    { {{{1, 64}, {1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND} },  // 输出
    &compileInfo);

// 3. ExecuteTestCase
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
