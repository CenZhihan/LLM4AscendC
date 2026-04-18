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

template <typename T, typename U>
__global__ __aicore__ void MainReducev2(__gm__ uint16_t* __restrict__ dstGm, __gm__ uint16_t* __restrict__ src0Gm,
    __gm__ uint16_t* __restrict__ src1Gm, int16_t dataSize1, int16_t dataSize2, int16_t dataSize3)
{
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<U> input1Global;
    GlobalTensor<T> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize1);
    input1Global.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(src1Gm), dataSize2);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize3);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize1 * sizeof(T));
    LocalTensor<T> input0Local = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize2 * sizeof(U));
    LocalTensor<U> input1Local = tbuf1.Get<U>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize3 * sizeof(T));
    LocalTensor<T> output_local = tbuf2.Get<T>();

    DataCopy(input0Local, input0Global, dataSize1);
    DataCopy(input1Local, input1Global, dataSize2);


    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);


    for (int i = 0; i < dataSize2; i++) {
        input1Local.SetValue(i, 65535);
    }
    uint64_t rsvdCnt = 0;
    GatherMask(output_local, input0Local, input1Local, false, 0, { 1, 1, 0, 0 }, rsvdCnt);
    GatherMask(output_local, input0Local, 1, false, 0, { 1, 1, 0, 0 }, rsvdCnt);
    GatherMask(output_local, input0Local, input1Local, true, 64, { 1, 1, 0, 0 }, rsvdCnt);

    uint64_t gatherMaskCount = GetGatherMaskRemainCount();
    EXPECT_EQ(rsvdCnt, gatherMaskCount);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, output_local, dataSize3);

    pipe_barrier(PIPE_ALL);
}

struct Reducev2TestParams {
    uint8_t typeBytes;
    uint16_t dataSize1;
    uint16_t dataSize2;
    uint16_t dataSize3;
    void (*CalFunc)(uint16_t*, uint16_t*, uint16_t*, int16_t, int16_t, int16_t);
};

class Reducev2Testsuite : public testing::Test, public testing::WithParamInterface<Reducev2TestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_REDUCEV2, Reducev2Testsuite,
    ::testing::Values(Reducev2TestParams { 2, 128, 16, 128, MainReducev2<uint16_t, uint16_t> },
    Reducev2TestParams { 4, 64, 8, 32, MainReducev2<uint32_t, uint32_t> }));

TEST_P(Reducev2Testsuite, Reducev2TestCase)
{
    auto param = GetParam();
    uint16_t dstGm[param.dataSize1 * param.typeBytes];
    uint16_t src0Gm[param.dataSize2 * param.typeBytes];
    uint16_t src1Gm[param.dataSize3 * param.typeBytes];

    param.CalFunc(dstGm, src0Gm, src1Gm, param.dataSize1, param.dataSize2, param.dataSize3);

    for (int i = 0; i < param.dataSize2; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}