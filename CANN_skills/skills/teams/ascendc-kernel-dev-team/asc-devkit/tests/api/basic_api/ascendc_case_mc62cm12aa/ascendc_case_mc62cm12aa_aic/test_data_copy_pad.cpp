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
#define private public
#define protect public
#include "kernel_operator.h"
// #include "api_check/kernel_cpu_check.h"

using namespace std;
using namespace AscendC;

struct TestDataCopyPadParams {
    uint32_t dataSize;
    uint32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, int32_t);
};

class TestDataCopyPadSuite : public testing::Test, public testing::WithParamInterface<TestDataCopyPadParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

template<typename T>
void main_data_copy_pad_kernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;

    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    GlobalTensor<half> testGlobal;
    testGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dstGm), dataSize);

    InitOutput(testGlobal, dataSize * sizeof(T) / sizeof(half));
    LocalTensor<T> inputLocal;
    TBuffAddr tbuf;
    tbuf.logicPos = (uint8_t)TPosition::A1;
    inputLocal.SetAddr(tbuf);
    inputLocal.InitBuffer(0, dataSize);
    DataCopyExtParams copyParams{1, 20 * sizeof(T), 0, 0, 0};
    DataCopyParams copyParams2{1, 32 * sizeof(T), 0, 0};
    DataCopyPadParams datacopyPadParams{false, 0, 0, 0};
    if constexpr(sizeof(T) == 8) {
        DataCopyPadExtParams<T> padParams{true, 0, 1, 0};
        DataCopyPad<T, PaddingMode::Normal>(inputLocal, srcGlobal, copyParams2, datacopyPadParams);
        DataCopyPad<T, PaddingMode::Normal>(inputLocal, srcGlobal, copyParams, padParams);
    } else {
        DataCopyPadExtParams<T> padParams{true, 0, 1, 3};
        DataCopyPad<T, PaddingMode::Normal>(inputLocal, srcGlobal, copyParams2, datacopyPadParams);
        DataCopyPad<T, PaddingMode::Normal>(inputLocal, srcGlobal, copyParams, padParams);
    }
}

INSTANTIATE_TEST_CASE_P(TEST_DATA_COPY_PAD, TestDataCopyPadSuite,
    ::testing::Values(TestDataCopyPadParams { 64, 4, main_data_copy_pad_kernel<int32_t> },
    TestDataCopyPadParams { 64, 2, main_data_copy_pad_kernel<int16_t> },
    TestDataCopyPadParams { 64, 4, main_data_copy_pad_kernel<float> },
    TestDataCopyPadParams { 64, 2, main_data_copy_pad_kernel<half> },
    TestDataCopyPadParams { 64, 1, main_data_copy_pad_kernel<int8_t> },
    TestDataCopyPadParams { 64, 8, main_data_copy_pad_kernel<int64_t> },
    TestDataCopyPadParams { 64, 1, main_data_copy_pad_kernel<uint8_t> }));

TEST_P(TestDataCopyPadSuite, TestDataCopyPadCases)
{
    SetGCoreType(2);
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.typeSize];
    uint8_t dstGm[param.dataSize * param.typeSize];
    param.cal_func(srcGm, dstGm, param.dataSize);
    SetGCoreType(0);
}
