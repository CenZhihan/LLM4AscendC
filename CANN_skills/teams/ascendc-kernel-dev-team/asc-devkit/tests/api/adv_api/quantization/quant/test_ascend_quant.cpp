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

constexpr AscendQuantConfig ASCEND_QUANT_DEFAULT_CFG = {256, 0, 0, 512};

template <typename T>
void AscendQuantKernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize,
    bool isQuantConfig)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<int8_t> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, AlignUp(dataSize * sizeof(int8_t), ONE_BLK_SIZE));
    LocalTensor<int8_t> outputLocal = tbuf2.Get<int8_t>();

    DataCopy(inputLocal, inputGlobal, dataSize);

    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    if (isQuantConfig) {
        AscendQuant<T, false, ASCEND_QUANT_DEFAULT_CFG>(outputLocal, inputLocal, 2, 0.9, dataSize);
    } else {
        AscendQuant(outputLocal, inputLocal, 2, 0.9, dataSize);
    }

    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    DataCopy(outputGlobal, outputLocal, dataSize);
    PipeBarrier<PIPE_ALL>();
}

struct AscendQuantTestParams {
    int32_t dataSize;
    int32_t dataBitSize;
    bool isQuantConfig;
    void (*calFunc)(uint8_t*, uint8_t*, int32_t);
};

class AscendQuantTestsuite : public testing::Test, public testing::WithParamInterface<AscendQuantTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_ASCEND_QUANT, AscendQuantTestsuite,
    ::testing::Values(AscendQuantTestParams { 256, 2, false, AscendQuantKernel<half> },
    AscendQuantTestParams { 256, 4, false, AscendQuantKernel<float> },
    AscendQuantTestParams { 256, 2, true, AscendQuantKernel<half> },
    AscendQuantTestParams { 256, 4, true, AscendQuantKernel<float> }));

TEST_P(AscendQuantTestsuite, AscendQuantTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.dataBitSize];
    uint8_t dstGm[param.dataSize] = { 0 };

    param.calFunc(srcGm, dstGm, param.dataSize, param.isQuantConfig);
    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
