#include "gtest/gtest.h"
#define private public
#define protected public
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected

// NodeDefBuilder
#define CREATE_NODEDEF(shapes, data_types, datas)                    \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "OpName", "OpName")               \
        .Input({"input1", data_types[0], shapes[0], datas[0]})       \
        .Output({"output", data_types[1], shapes[1], datas[1]});

// 执行
RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

// 验证
bool compare = CompareResult(output, expected, count);
