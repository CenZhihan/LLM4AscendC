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
__global__ __aicore__ void MainNchwconv(__gm__ uint16_t* __restrict__ dstGm, __gm__ uint16_t* __restrict__ srcGm,
    int32_t dataSize, const TransDataTo5HDParams& nchwconvParams, const int32_t width)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::CO2> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    TBuf<TPosition::CO2> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T) * 2);
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    TBuf<TPosition::CO2> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(uint64_t));
    LocalTensor<uint64_t> src = tbuf2.Get<uint64_t>();

    TBuf<TPosition::CO2> tbuf3;
    tpipe.InitBuffer(tbuf3, dataSize * sizeof(uint64_t));
    LocalTensor<uint64_t> dst = tbuf3.Get<uint64_t>();

    DataCopy(inputLocal, inputGlobal, dataSize);

    auto eventIdMTE2ToS = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
    SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);

    for (int i = 0; i < 16; i++) {
        dst.SetValue(i, (uint64_t)outputLocal[width * i].GetPhyAddr());
        src.SetValue(i, (uint64_t)inputLocal[width * i].GetPhyAddr());
    }

    TransDataTo5HD<T>(dst, src, nchwconvParams);

    auto eventIdVToMTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    DataCopy(outputGlobal, outputLocal, dataSize * 2);

    pipe_barrier(PIPE_ALL);
}

struct NchwconvTestParams {
    int32_t dataSize;
    int32_t typeBytes;
    bool dstHigh;
    bool srcHigh;
    const uint8_t repTimes;
    const uint16_t dstRep;
    const uint16_t srcRep;
    void (*cal_func)(uint16_t*, uint16_t*, int32_t, const TransDataTo5HDParams&, const int32_t);
    const int32_t width;
};

class NchwconvTestSuite : public testing::Test, public testing::WithParamInterface<NchwconvTestParams> {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(2);
    }
    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_NCHWCONV, NchwconvTestSuite,
    ::testing::Values(NchwconvTestParams { 256, 2, false, false, 1, 16, 16, MainNchwconv<half>, 16 },
    NchwconvTestParams { 512, 2, false, false, 2, 16, 16, MainNchwconv<uint16_t>, 16 },
    NchwconvTestParams { 512, 2, false, false, 2, 1, 1, MainNchwconv<int16_t>, 32 },
    NchwconvTestParams { 512, 1, true, true, 1, 16, 16, MainNchwconv<int8_t>, 32 },
    NchwconvTestParams { 512, 1, true, false, 1, 16, 16, MainNchwconv<int8_t>, 32 },
    NchwconvTestParams { 1024, 1, false, false, 2, 16, 16, MainNchwconv<uint8_t>, 32 },
    NchwconvTestParams { 1024, 1, false, true, 2, 1, 1, MainNchwconv<uint8_t>, 64 }));

TEST_P(NchwconvTestSuite, NchwconvTestCase)
{
    auto param = GetParam();
    TransDataTo5HDParams nchwconvParams;
    nchwconvParams.dstHighHalf = param.dstHigh;
    nchwconvParams.srcHighHalf = param.srcHigh;
    nchwconvParams.repeatTimes = param.repTimes;
    nchwconvParams.dstRepStride = param.dstRep;
    nchwconvParams.srcRepStride = param.srcRep;

    uint16_t dstGm[param.dataSize * param.typeBytes] = {0};
    uint16_t srcGm[param.dataSize * param.typeBytes] = {0};

    param.cal_func(dstGm, srcGm, param.dataSize, nchwconvParams, param.width);

    for (int i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}