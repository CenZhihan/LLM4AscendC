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
#include <string>
#include <vector>

#include "aclnn/aclnn_base.h"
#include "aclnn/acl_meta.h"
#include "acl/acl_rt.h"
#include "aclnn_add_custom_template.h"

#define CHECK_ACL(expr)                                                                                 \
    do {                                                                                                \
        auto __ret = (expr);                                                                            \
        int32_t __code = static_cast<int32_t>(__ret);                                                   \
        if (__code != 0) {                                                                              \
            fprintf(stderr, "[ERROR] %s failed at %s:%d, ret=%d\n", #expr, __FILE__, __LINE__, __code); \
        }                                                                                               \
    } while (0)

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<float> {
    static constexpr aclDataType kAclType = ACL_FLOAT;
    static float ToFloat(float v) { return v; }
    static float FromFloat(float v) { return v; }
};

template <>
struct TypeTraits<aclFloat16> {
    static constexpr aclDataType kAclType = ACL_FLOAT16;
    static float ToFloat(aclFloat16 v) { return aclFloat16ToFloat(v); }
    static aclFloat16 FromFloat(float v) { return aclFloatToFloat16(v); }
};

template <typename T>
int32_t RunOnce(const std::vector<int64_t>& shape)
{
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_ACL(aclnnInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    const int64_t elementCount = shape[0] * shape[1];
    const size_t bufferSize = elementCount * sizeof(T);

    void* input0DeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&input0DeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* input0 = aclCreateTensor(shape.data(), shape.size(), TypeTraits<T>::kAclType, nullptr, 0, ACL_FORMAT_ND,
                                        shape.data(), shape.size(), input0DeviceMem);

    void* input1DeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&input1DeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* input1 = aclCreateTensor(shape.data(), shape.size(), TypeTraits<T>::kAclType, nullptr, 0, ACL_FORMAT_ND,
                                        shape.data(), shape.size(), input1DeviceMem);

    void* output0DeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&output0DeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* output0 = aclCreateTensor(shape.data(), shape.size(), TypeTraits<T>::kAclType, nullptr, 0, ACL_FORMAT_ND,
                                         shape.data(), shape.size(), output0DeviceMem);

    std::vector<T> input0HostData(elementCount, TypeTraits<T>::FromFloat(1.0f));
    std::vector<T> input1HostData(elementCount, TypeTraits<T>::FromFloat(2.0f));
    std::vector<T> output0HostData(elementCount, TypeTraits<T>::FromFloat(0.0f));
    std::vector<T> goldenData(elementCount, TypeTraits<T>::FromFloat(3.0f));

    CHECK_ACL(aclrtMemcpy(input0DeviceMem, bufferSize, input0HostData.data(),
                          bufferSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(input1DeviceMem, bufferSize, input1HostData.data(),
                          bufferSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnAddCustomTemplateGetWorkspaceSize(input0, input1, output0, &workspaceSize, &executor));
    void* workspaceDeviceMem = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspaceDeviceMem, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL(aclnnAddCustomTemplate(workspaceDeviceMem, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(output0HostData.data(), bufferSize, output0DeviceMem,
                          bufferSize, ACL_MEMCPY_DEVICE_TO_HOST));

    printf("result is:\n");
    const int64_t previewCount = std::min<int64_t>(elementCount, 10);
    for (int64_t i = 0; i < previewCount; i++) { printf("%.6f ", TypeTraits<T>::ToFloat(output0HostData[i])); }
    printf("\ntest %s\n", std::equal(output0HostData.begin(), output0HostData.end(), goldenData.begin()) ? "pass" : "failed");

    aclDestroyTensor(input0);
    aclDestroyTensor(input1);
    aclDestroyTensor(output0);
    CHECK_ACL(aclrtFree(input0DeviceMem));
    CHECK_ACL(aclrtFree(input1DeviceMem));
    CHECK_ACL(aclrtFree(output0DeviceMem));
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtFree(workspaceDeviceMem));
    }
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclnnFinalize());
    return 0;
}

int32_t main(int32_t argc, char** argv)
{
    const std::string dtype = (argc > 1) ? argv[1] : "float16";
    std::vector<int64_t> shape = {8, 128};
    if (argc >= 3) {
        shape[1] = std::stoll(argv[2]);
    }
    if (shape[1] != 128 && shape[1] != 2048) {
        fprintf(stderr, "ERROR: dim1 only supports 128 or 2048, got %ld\n", shape[1]);
        fprintf(stderr, "Usage: %s [float|float16] [128|2048]\n", argv[0]);
        return 1;
    }
    if (dtype == "float") {
        return RunOnce<float>(shape);
    }
    if (dtype == "float16") {
        return RunOnce<aclFloat16>(shape);
    }
    fprintf(stderr, "Usage: %s [float|float16] [128|2048]\n", argv[0]);
    return 1;
}
