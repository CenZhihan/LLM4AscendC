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

constexpr AscendQuantConfig ASCEND_QUANT_DEFAULT_CFG = {2048, 512, 512, 4096};

template <typename T>
void AscendQuantPerChannelKernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm,
    __gm__ uint8_t* __restrict__ scaleGm, __gm__ uint8_t* __restrict__ offsetGm, __gm__ int32_t dataSize,
    bool isQuantConfig)
{
    TPipe tpipe;
    GlobalTensor<half> inputGlobal;
    GlobalTensor<half> scaleGlobal;
    GlobalTensor<half> offsetGlobal;
    GlobalTensor<int8_t> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize * 8 * sizeof(half));
    scaleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scaleGm), dataSize * sizeof(half));
    offsetGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(offsetGm), dataSize * sizeof(half));
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(dstGm), dataSize * 8);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * 8 * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> scaleLocal = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<T> offsetLocal = tbuf2.Get<T>();

    TBuf<TPosition::VECCALC> tbuf3;
    tpipe.InitBuffer(tbuf3, dataSize * 8 * sizeof(int8_t));
    LocalTensor<int8_t> outputLocal = tbuf3.Get<int8_t>();

    DataCopy(inputLocal, inputGlobal, dataSize * 8);
    DataCopy(scaleLocal, scaleGlobal, dataSize);
    DataCopy(offsetLocal, offsetGlobal, dataSize);

    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    if (isQuantConfig) {
        AscendQuant<T, false, ASCEND_QUANT_DEFAULT_CFG>(outputLocal, inputLocal, scaleLocal, offsetLocal);
    } else {
        AscendQuant(outputLocal, inputLocal, scaleLocal, offsetLocal);
    }

    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    DataCopy(outputGlobal, outputLocal, dataSize);
    PipeBarrier<PIPE_ALL>();
}

struct AscendQuantPerChannelTestParams {
    int32_t dataSize;
    bool isQuantConfig;
    void (*calFunc)(uint8_t*, uint8_t*, uint8_t*, uint8_t*, int32_t);

};

class AscendQuantPerChannelTestsuite : public testing::Test, public testing::WithParamInterface<AscendQuantPerChannelTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_ASCEND_QUANT_PER_CHANNEL, AscendQuantPerChannelTestsuite,
    ::testing::Values(AscendQuantPerChannelTestParams { 256, false, AscendQuantPerChannelKernel<half> },
    AscendQuantPerChannelTestParams { 1024, false, AscendQuantPerChannelKernel<half> },
    AscendQuantPerChannelTestParams { 256, true, AscendQuantPerChannelKernel<half> },
    AscendQuantPerChannelTestParams { 1024, true, AscendQuantPerChannelKernel<half> }));

TEST_P(AscendQuantPerChannelTestsuite, AscendQuantPerChannelTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * 8 * sizeof(half)] = {0};
    uint8_t dstGm[param.dataSize * 8] = {0};
    uint8_t scaleGm[param.dataSize * sizeof(half)] = {0};
    uint8_t offsetGm[param.dataSize * sizeof(half)] = {0};

    param.calFunc(srcGm, dstGm, scaleGm, offsetGm, param.dataSize, param.isQuantConfig);
    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
