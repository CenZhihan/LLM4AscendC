#include "node_def_builder.h"

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "OpName", "OpName")               \
        .Input({"input1", data_types[0], shapes[0], datas[0]})       \
        .Output({"output", data_types[1], shapes[1], datas[1]});

RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
