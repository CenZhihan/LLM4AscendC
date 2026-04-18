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

using namespace std;
using namespace AscendC;

template <typename T>
void TanhKernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    SetVectorMask<uint8_t, MaskMode::NORMAL>(256);

    DataCopy(inputLocal, inputGlobal, dataSize);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetVectorMask<uint8_t, MaskMode::NORMAL>(128);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    Tanh(outputLocal, inputLocal, dataSize);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    PipeBarrier<PIPE_ALL>();
}

struct TanhTestParams {
    int32_t dataSize;
    int32_t dataBitSize;
    void (*calFunc)(uint8_t*, uint8_t*, int32_t);
};

class TanhTestsuite : public testing::Test, public testing::WithParamInterface<TanhTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_TANH, TanhTestsuite,
    ::testing::Values(TanhTestParams { 256, 2, TanhKernel<half> }, TanhTestParams { 256, 4, TanhKernel<float> }));

TEST_P(TanhTestsuite, TanhTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.dataBitSize] = {0};
    uint8_t dstGm[param.dataSize * param.dataBitSize] = {0};

    param.calFunc(srcGm, dstGm, param.dataSize);
    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
