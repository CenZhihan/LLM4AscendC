
#include "matmul_with_diagonal_matrices_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    (void)ascendcPlatform;
    auto shapeA = context->GetInputTensor(0)->GetOriginShape();
    auto shapeB = context->GetInputTensor(1)->GetOriginShape();

    uint32_t rows = static_cast<uint32_t>(shapeA.GetDim(0));
    uint32_t cols = static_cast<uint32_t>(shapeB.GetDim(1));
    uint32_t tileLength = 8192;

    // correctness-first: force single-core to avoid row-chunk alignment pitfalls
    uint32_t coreNum = 1;

    MatmulWithDiagonalMatricesCustomTilingData tiling;
    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_tileLength(tileLength);

    context->SetBlockDim(coreNum);
    context->SetTilingKey(0);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 16 * 1024 * 1024;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    *context->GetOutputShape(0) = *context->GetInputShape(1);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(1));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class MatmulWithDiagonalMatricesCustom : public OpDef {
public:
    explicit MatmulWithDiagonalMatricesCustom(const char *name) : OpDef(name)
    {
        this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("B").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("C").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(MatmulWithDiagonalMatricesCustom);
} // namespace ops

