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
#define protected public
#include "kernel_operator.h"
// #include "model/model_factory.h"
#include "common.h"

using namespace std;
using namespace AscendC;

class TEST_TERNARY : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

template <typename T, typename U>
__global__ __aicore__ void MainAxpy(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm,
    int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<U> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::CO2> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(U));
    LocalTensor<U> inputLocal = tbuf.Get<U>();

    TBuf<TPosition::CO2> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    U scalarValue(1.0);

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Axpy(outputLocal, inputLocal, scalarValue, dataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    pipe_barrier(PIPE_ALL);
}

struct AxpyTestParams {
    int32_t dataSize;
    int32_t dstTypeBytes;
    int32_t srcTypeBytes;
    uint64_t finalMaskHigh;
    uint64_t finalMaskLow;
    void (*cal_func)(uint8_t*, uint8_t*, int32_t);
};

class AxpyTestsuite : public testing::Test, public testing::WithParamInterface<AxpyTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_AXPY, AxpyTestsuite,
    ::testing::Values(AxpyTestParams { 32, 2, 2, 0, GenerateMask(32), MainAxpy<half, half> },
    AxpyTestParams { 32, 4, 2, 0, GenerateMask(32), MainAxpy<float, half> },
    AxpyTestParams { 32, 4, 4, 0, GenerateMask(32), MainAxpy<float, float> },
    AxpyTestParams { 64, 2, 2, 0, GenerateMask(64), MainAxpy<half, half> },
    AxpyTestParams { 64, 4, 2, 0, GenerateMask(64), MainAxpy<float, half> },
    AxpyTestParams { 64, 4, 4, 0, GenerateMask(64), MainAxpy<float, float> },
    AxpyTestParams { 128, 2, 2, GenerateMask(64), GenerateMask(64), MainAxpy<half, half> },
    AxpyTestParams { 128, 4, 2, 0, GenerateMask(64), MainAxpy<float, half> },
    AxpyTestParams { 128, 4, 4, 0, GenerateMask(64), MainAxpy<float, float> },
    AxpyTestParams { 288, 2, 2, 0, GenerateMask(32), MainAxpy<half, half> },
    AxpyTestParams { 288, 4, 2, 0, GenerateMask(32), MainAxpy<float, half> },
    AxpyTestParams { 288, 4, 4, 0, GenerateMask(32), MainAxpy<float, float> }));

TEST_P(AxpyTestsuite, AxpyLevel2TestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.srcTypeBytes];
    uint8_t dstGm[param.dataSize * param.dstTypeBytes];

    param.cal_func(dstGm, srcGm, param.dataSize);
    for (int32_t i = 0; i < param.dataSize * param.srcTypeBytes; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}