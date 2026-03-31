/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include "kernel_operator.h"
// #include "api_check/kernel_cpu_check.h"
#include "tikicpulib.h"
 
using namespace std;
using namespace AscendC;
 
struct TestDataTypeParams {
    uint32_t dataSize;
    uint32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, int32_t);
};
 
class TestDataTypeSuite : public testing::Test, public testing::WithParamInterface<TestDataTypeParams> {
protected:
    void SetUp() {
        SetGCoreType(1);
    }
    void TearDown() {
        SetGCoreType(0);
    }
};
 
template<typename T>
void MainDataTypeKernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
 
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);
 
    if constexpr(IsSameType<T, float>::value) {
        test_encoding();
    }
    const float fp8e5m2 = 0.00000762939453125; // 0x37000000
    const float fp8e4m3 = 0.0009765625; // 0x3a800000
    float8_e5m2::Fp8e5m2T tmp1(fp8e5m2);
    float8_e4m3::Fp8e4m3T tmp2(fp8e4m3);
}
 
INSTANTIATE_TEST_CASE_P(TEST_DATA_COPY_AIC, TestDataTypeSuite,
    ::testing::Values(
        TestDataTypeParams { 64, 4, MainDataTypeKernel<float> }
    ));
 
TEST_P(TestDataTypeSuite, TestDataCopyPadCases)
{
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.typeSize];
    uint8_t dstGm[param.dataSize * param.typeSize];
    param.cal_func(srcGm, dstGm, param.dataSize);
}