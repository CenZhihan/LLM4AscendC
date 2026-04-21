#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

// InferShapeContextPara
gert::InfershapeContextPara infershapeContextPara("Abs",
    { {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND} },  // 输入
    { {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND} });               // 输出（空shape待推导）

// ExecuteTestCase
std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4}};
ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
