
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/matrix/matmul_tiling.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto* aShape = context->GetInputShape(0);
    auto* bShape = context->GetInputShape(1);

    // A: [M, K], B: [N, K], output C: [M, N]
    const auto& aDims = aShape->GetOriginShape().GetDims();
    const auto& bDims = bShape->GetOriginShape().GetDims();
    if (aDims.size() != 2 || bDims.size() != 2) {
        return ge::GRAPH_FAILED;
    }
    int64_t M = aDims[0];
    int64_t K = aDims[1];
    int64_t N = bDims[0]; // B is [N, K], we'll use transpose(B) in kernel

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform);

    tiling.SetDim(1);
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    // B is provided as [N, K]; the kernel will set transposeB=true in SetTensorB
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    tiling.SetShape(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    tiling.SetSingleShape(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    tiling.SetOrgShape(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1);

    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    if (ret != 0) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(tiling.GetCoreNum());

    // Save raw TCubeTiling bytes to runtime
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* a_shape = context->GetInputShape(0); // [M, K]
    const gert::Shape* b_shape = context->GetInputShape(1); // [N, K]
    if (a_shape->GetDimNum() != 2 || b_shape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    int64_t M = a_shape->GetDim(0);
    int64_t N = b_shape->GetDim(0);
    gert::Shape* c_shape = context->GetOutputShape(0);
    c_shape->SetDimNum(2);
    c_shape->SetDim(0, M);
    c_shape->SetDim(1, N);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // A/B are float16, C is float32 accumulation
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class MatmulWithTransposedBCustom : public OpDef {
public:
    explicit MatmulWithTransposedBCustom(const char* name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(MatmulWithTransposedBCustom);
}
