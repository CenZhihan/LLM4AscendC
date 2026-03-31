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

template <typename T>
__global__ __aicore__ void MainTranspose(__gm__ uint16_t* __restrict__ dstGm, __gm__ uint16_t* __restrict__ srcGm,
    int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> input_global;
    GlobalTensor<T> output_global;
    input_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::CO2> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> input_local = tbuf.Get<T>();

    TBuf<TPosition::CO2> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> output_local = tbuf1.Get<T>();


    DataCopy(input_local, input_global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Transpose(output_local, input_local);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(output_global, output_local, dataSize);

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