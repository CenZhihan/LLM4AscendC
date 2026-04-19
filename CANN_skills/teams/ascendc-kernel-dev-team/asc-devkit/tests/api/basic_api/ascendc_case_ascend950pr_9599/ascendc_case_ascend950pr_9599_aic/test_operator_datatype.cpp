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
        BufIdTracker::GetInstance().Reset();
    }
    void TearDown() {
        SetGCoreType(0);
    }
};
 
template<typename T, bool USE_TOFLOAT = true>
void MainDataTypeKernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
 
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);
 
    if constexpr(IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value) {
        T value;
        value.val = 1;
        float res = static_cast<float>(value);
        float res1;
        if constexpr (USE_TOFLOAT) 
            res1 = value.ToFloat();
        else
            res1 = Cast<T, float>(value);
        T res2{res1};
    }
    if constexpr(IsSameType<T, hifloat8_t>::value) {
        T value;
        value.val = -1;
        float res = static_cast<float>(value);
        float res1;
        if constexpr (USE_TOFLOAT) 
            res1 = value.ToFloat();
        else
            res1 = Cast<T, float>(value);
        T res2{res1};
        float x = 0.5f;
        T res3{x};
        float y = 8.0f;
        T res4{y};
    }

    if constexpr(IsSameType<T, fp4x2_e1m2_t>::value || IsSameType<T, fp4x2_e2m1_t>::value) {
        T value;
        value.val = 1;
        bfloat16_t res = static_cast<bfloat16_t>(value);
        T res2{res};
    }
    if constexpr(IsSameType<T, half>::value) {
        T value0, value1;
        value0++;
        value0--;
        ++value0;
        --value0;
        bool tmp = value0 && value1;
        tmp = value0 || value1;
    }
    if constexpr(IsSameType<T, float>::value) {
        test_encoding();
    }
    if constexpr(IsSameType<T, bfloat16_t>::value) {
        fp4x2_e1m2_t value3, value4;
        value3.val = 1;
        value4.val = 2;
        bfloat16_t res3 = bfloat16_t(value3);
        bfloat16_t res4 = bfloat16_t(value4);
        bfloat16_t res5 = res3 - res4;
        res5 -= res3;
    }
}
 
INSTANTIATE_TEST_CASE_P(TEST_DATA_COPY_AIC_TOFLOAT, TestDataTypeSuite,
    ::testing::Values(
        TestDataTypeParams { 64, 4, MainDataTypeKernel<fp8_e4m3fn_t, true> },
        TestDataTypeParams { 64, 4, MainDataTypeKernel<fp8_e5m2_t, true> },
        TestDataTypeParams { 64, 4, MainDataTypeKernel<hifloat8_t, true> },
        TestDataTypeParams { 64, 4, MainDataTypeKernel<fp4x2_e1m2_t, true> },
        TestDataTypeParams { 64, 4, MainDataTypeKernel<fp4x2_e2m1_t, true> },
        TestDataTypeParams { 64, 4, MainDataTypeKernel<half, true> },
        TestDataTypeParams { 64, 4, MainDataTypeKernel<float, true> },
        TestDataTypeParams { 64, 4, MainDataTypeKernel<bfloat16_t, true> }
    ));

INSTANTIATE_TEST_CASE_P(TEST_DATA_COPY_AIC_CAST, TestDataTypeSuite,
    ::testing::Values(
        TestDataTypeParams { 64, 4, MainDataTypeKernel<fp8_e4m3fn_t, false> },
        TestDataTypeParams { 64, 4, MainDataTypeKernel<fp8_e5m2_t, false> },
        TestDataTypeParams { 64, 4, MainDataTypeKernel<hifloat8_t, false> },
        TestDataTypeParams { 64, 4, MainDataTypeKernel<fp4x2_e1m2_t, false> },
        TestDataTypeParams { 64, 4, MainDataTypeKernel<fp4x2_e2m1_t, false> },
        TestDataTypeParams { 64, 4, MainDataTypeKernel<half, false> },
        TestDataTypeParams { 64, 4, MainDataTypeKernel<float, false> },
        TestDataTypeParams { 64, 4, MainDataTypeKernel<bfloat16_t, false> }
    ));

TEST_P(TestDataTypeSuite, TestDataCopyPadCases)
{
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.typeSize];
    uint8_t dstGm[param.dataSize * param.typeSize];
    param.cal_func(srcGm, dstGm, param.dataSize);
}