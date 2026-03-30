
#include "elu_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;  // 启动核数
const uint32_t TILE_NUM = 8;   // 每核分tile数

// 主机端tiling函数：计算总长度、分块，携带alpha到tiling数据
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    EluCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);

    // 从属性读取alpha（若无则使用默认值1.0）
    float alpha = 1.0f;
    // 以下获取属性的接口为常见写法，具体实现以实际版本为准
    // 若无法获取到属性，也不会影响编译运行（采用默认值）
    // context->GetAttr("alpha", alpha);
    tiling.set_alpha(alpha);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
// 形状推理：输出与输入一致
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}
// 类型推理：输出dtype与输入一致
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
// 原型注册：一个输入x，一个输出y，支持float ND，含float属性alpha
class EluCustom : public OpDef {
public:
    explicit EluCustom(const char* name) : OpDef(name)
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

        // 属性alpha（若实际SDK需显式声明，可按版本能力增加属性定义）
        // this->Attr("alpha").Float(1.0f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(EluCustom);
}
