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
// #include "model/model_factory.h"
#include "common.h"

using namespace std;
using namespace AscendC;

template <typename T>
__global__ __aicore__ void MainTranspose(__gm__ uint16_t* __restrict__ dstGm, __gm__ uint16_t* __restrict__ srcGm,
    int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECIN> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    TBuf<TPosition::VECOUT> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Transpose(outputLocal, inputLocal);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    pipe_barrier(PIPE_ALL);
}

struct TransposeTestParams {
    int32_t dataSize;
    int32_t typeBytes;
    void (*cal_func)(uint16_t*, uint16_t*, int32_t);
};

class TransposeTestsuite : public testing::Test, public testing::WithParamInterface<TransposeTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_TRANSPOSE, TransposeTestsuite,
    ::testing::Values(TransposeTestParams { 256, 2, MainTranspose<half> },
    TransposeTestParams { 256, 2, MainTranspose<int16_t> }, TransposeTestParams { 256, 2, MainTranspose<uint16_t> }));

TEST_P(TransposeTestsuite, TransposeTestCase)
{
    auto param = GetParam();

    uint16_t dstGm[param.dataSize * param.typeBytes];
    uint16_t srcGm[param.dataSize * param.typeBytes];

    param.cal_func(dstGm, srcGm, param.dataSize);

    for (int i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}