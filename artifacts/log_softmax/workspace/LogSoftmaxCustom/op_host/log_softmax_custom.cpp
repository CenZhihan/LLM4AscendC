
#include "log_softmax_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LogSoftmaxCustomTilingData tiling;

    // 输入形状
    const gert::Shape* in_shape_desc = context->GetInputShape(0);
    ge::Shape srcShape = in_shape_desc->GetOriginShape();

    // 数据类型大小：float32 -> 4 bytes
    const uint32_t dtypeSize = 4;

    // 选择满足正确性的最小临时空间
    const uint32_t localWorkSpaceSize = AscendC::GetLogSoftMaxMinTmpSize(srcShape, dtypeSize, false);

    // 获取LogSoftMax所需的tiling参数
    AscendC::LogSoftMaxTilingFunc(srcShape, dtypeSize, localWorkSpaceSize, tiling.logSoftmaxTilingData);

    // 为了内核GM张量设置长度（元素数）
    uint32_t totalLength = in_shape_desc->GetOriginShape().GetShapeSize();
    tiling.set_totalLength(totalLength);

    // 设置BlockDim（示例值）
    context->SetBlockDim(8);

    // 无需额外workspace
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge


namespace ops {
class LogSoftmaxCustom : public OpDef {
public:
    explicit LogSoftmaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LogSoftmaxCustom);
} // namespace ops
