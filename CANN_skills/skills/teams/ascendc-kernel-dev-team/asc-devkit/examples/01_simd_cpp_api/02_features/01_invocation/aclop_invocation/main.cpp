/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "acl/acl_mdl.h"
#include "acl/acl_op.h"

#define CHECK_ACL(expr)                                                                                 \
    do {                                                                                                \
        auto __ret = (expr);                                                                            \
        int32_t __code = static_cast<int32_t>(__ret);                                                   \
        if (__code != 0) {                                                                              \
            fprintf(stderr, "[ERROR] %s failed at %s:%d, ret=%d\n", #expr, __FILE__, __LINE__, __code); \
        }                                                                                               \
    } while (0)

int32_t main(int32_t argc, char** argv)
{
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));
    CHECK_ACL(aclopSetModelDir("."));

    const char* opType = "AddCustom";

    std::vector<aclTensorDesc*> inputDesc;
    std::vector<aclTensorDesc*> outputDesc;
    const aclDataType dataType = ACL_FLOAT16;
    const aclFormat format = ACL_FORMAT_ND;
    const std::vector<int64_t> shape = {8, 2048};
    const int64_t elementCount = shape[0] * shape[1];
    const size_t bufferSize = elementCount * sizeof(aclFloat16);
    auto* input0 = aclCreateTensorDesc(dataType, shape.size(), shape.data(), format);
    auto* input1 = aclCreateTensorDesc(dataType, shape.size(), shape.data(), format);
    auto* output0 = aclCreateTensorDesc(dataType, shape.size(), shape.data(), format);
    inputDesc.push_back(input0);
    inputDesc.push_back(input1);
    outputDesc.push_back(output0);

    aclopAttr* opAttr = nullptr;
    opAttr = aclopCreateAttr();

    std::vector<aclDataBuffer*> inputBuffers;
    std::vector<aclDataBuffer*> outputBuffers;

    void* input0DeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&input0DeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto* input0Buffer = aclCreateDataBuffer(input0DeviceMem, bufferSize);
    inputBuffers.push_back(input0Buffer);

    void* input1DeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&input1DeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto* input1Buffer = aclCreateDataBuffer(input1DeviceMem, bufferSize);
    inputBuffers.push_back(input1Buffer);

    void* output0DeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&output0DeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto* output0Buffer = aclCreateDataBuffer(output0DeviceMem, bufferSize);
    outputBuffers.push_back(output0Buffer);

    std::vector<aclFloat16> input0HostData(elementCount, aclFloatToFloat16(1.0));
    std::vector<aclFloat16> input1HostData(elementCount, aclFloatToFloat16(2.0));
    std::vector<aclFloat16> output0HostData(elementCount, aclFloatToFloat16(0.0));
    std::vector<aclFloat16> goldenData(elementCount, aclFloatToFloat16(3.0));

    CHECK_ACL(aclrtMemcpy(input0DeviceMem, bufferSize, input0HostData.data(),
                          bufferSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(input1DeviceMem, bufferSize, input1HostData.data(),
                          bufferSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclopExecuteV2(opType, inputDesc.size(), inputDesc.data(), inputBuffers.data(),
                             outputDesc.size(), outputDesc.data(), outputBuffers.data(), opAttr, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(output0HostData.data(), bufferSize, output0DeviceMem,
                          bufferSize, ACL_MEMCPY_DEVICE_TO_HOST));

    printf("result is:\n");
    const int64_t previewCount = std::min<int64_t>(elementCount, 10);
    for (int64_t i = 0; i < previewCount; i++) { printf("%.1f ", aclFloat16ToFloat(output0HostData[i])); }
    printf("\ntest %s\n", std::equal(output0HostData.begin(), output0HostData.end(), goldenData.begin()) ? "pass" : "failed");

    CHECK_ACL(aclDestroyDataBuffer(input0Buffer));
    CHECK_ACL(aclDestroyDataBuffer(input1Buffer));
    CHECK_ACL(aclDestroyDataBuffer(output0Buffer));
    CHECK_ACL(aclrtFree(input0DeviceMem));
    CHECK_ACL(aclrtFree(input1DeviceMem));
    CHECK_ACL(aclrtFree(output0DeviceMem));
    aclDestroyTensorDesc(input0);
    aclDestroyTensorDesc(input1);
    aclDestroyTensorDesc(output0);
    aclopDestroyAttr(opAttr);
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}
