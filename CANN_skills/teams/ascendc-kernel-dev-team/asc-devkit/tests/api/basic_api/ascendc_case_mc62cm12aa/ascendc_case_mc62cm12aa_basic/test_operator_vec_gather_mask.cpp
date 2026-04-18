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
// #include "model/model_factory.h"
#include "common.h"

using namespace std;
using namespace AscendC;

template <typename T, typename U>
__global__ __aicore__ void MainReducev2(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ src1Gm, int16_t dataSize1, int16_t dataSize2, int16_t dataSize3)
{
    TPipe tpipe;
    GlobalTensor<T> input0_global;
    GlobalTensor<U> input1_global;
    GlobalTensor<T> output_global;
    input0_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize1);
    input1_global.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(src1Gm), dataSize2);
    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize3);

    LocalTensor<T> input0_local;
    TBuffAddr tbuf0;
    tbuf0.logicPos = (uint8_t)TPosition::VECCALC;
    input0_local.SetAddr(tbuf0);
    input0_local.InitBuffer(0, dataSize1);

    LocalTensor<U> input1_local;
    TBuffAddr tbuf1;
    tbuf1.logicPos = (uint8_t)TPosition::VECCALC;
    input1_local.SetAddr(tbuf1);
    input1_local.InitBuffer(dataSize1 * sizeof(T), dataSize2);

    LocalTensor<T> output_local;
    TBuffAddr tbuf2;
    tbuf2.logicPos = (uint8_t)TPosition::VECCALC;
    output_local.SetAddr(tbuf2);
    output_local.InitBuffer((dataSize1 + dataSize2) * sizeof(T), dataSize3);

    DataCopy(input0_local, input0_global, dataSize1);
    DataCopy(input1_local, input1_global, dataSize2);


    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);


    for (int i = 0; i < dataSize2; i++) {
        input1_local.SetValue(i, 65535);
    }
    uint64_t rsvdCnt = 0;
    if constexpr (sizeof(T) != 4) {
        GatherMask(output_local, input0_local, 1, false, 0, { 1, 1, 0, 0 }, rsvdCnt);
	GatherMask(output_local, input0_local, 2, false, 0, { 1, 1, 0, 0 }, rsvdCnt);
        GatherMask(output_local, input0_local, 3, false, 0, { 1, 1, 0, 0 }, rsvdCnt);
        GatherMask(output_local, input0_local, 4, false, 0, { 1, 1, 0, 0 }, rsvdCnt);
        GatherMask(output_local, input0_local, 5, false, 0, { 1, 1, 0, 0 }, rsvdCnt);
        GatherMask(output_local, input0_local, 6, false, 0, { 1, 1, 0, 0 }, rsvdCnt);
        GatherMask(output_local, input0_local, 7, false, 0, { 1, 1, 0, 0 }, rsvdCnt);
        GatherMask(output_local, input0_local, 1, true, 120, { 1, 1, 0, 0 }, rsvdCnt);
        GatherMask(output_local, input0_local, 2, true, 120, { 1, 1, 0, 0 }, rsvdCnt);
        GatherMask(output_local, input0_local, 3, true, 120, { 1, 1, 0, 0 }, rsvdCnt);
        GatherMask(output_local, input0_local, 4, true, 120, { 1, 1, 0, 0 }, rsvdCnt);
        GatherMask(output_local, input0_local, 5, true, 120, { 1, 1, 0, 0 }, rsvdCnt);
        GatherMask(output_local, input0_local, 6, true, 120, { 1, 1, 0, 0 }, rsvdCnt);
        GatherMask(output_local, input0_local, 7, true, 120, { 1, 1, 0, 0 }, rsvdCnt);
        GatherMask(output_local, input0_local, input1_local, false, 0, { 1, 1, 0, 0 }, rsvdCnt);
        GatherMask(output_local, input0_local, input1_local, true, 64, { 1, 1, 0, 0 }, rsvdCnt);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(output_global, output_local, dataSize3);

    pipe_barrier(PIPE_ALL);
}

struct Reducev2TestParams {
    uint8_t typeBytes;
    uint16_t dataSize1;
    uint16_t dataSize2;
    uint16_t dataSize3;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, int16_t, int16_t, int16_t);
};

class Reducev2Testsuite : public testing::Test, public testing::WithParamInterface<Reducev2TestParams> {
protected:
    void SetUp()
    {
        SetGCoreType(2);
    }
    void TearDown()
    {
        SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_REDUCEV2, Reducev2Testsuite,
    ::testing::Values(
    Reducev2TestParams { 1, 256, 32, 256, MainReducev2<uint8_t, uint8_t> },
    Reducev2TestParams { 1, 256, 32, 256, MainReducev2<int8_t, uint8_t> },
    Reducev2TestParams { 2, 128, 16, 128, MainReducev2<half, uint16_t> },
    Reducev2TestParams { 2, 128, 16, 128, MainReducev2<uint16_t, uint16_t> },
    Reducev2TestParams { 2, 128, 16, 128, MainReducev2<bfloat16_t, uint16_t> },
    Reducev2TestParams { 2, 128, 16, 128, MainReducev2<int16_t, uint16_t> },
    Reducev2TestParams { 4, 64, 8, 64, MainReducev2<float, uint32_t> },
    Reducev2TestParams { 4, 64, 8, 64, MainReducev2<uint32_t, uint32_t> },
    Reducev2TestParams { 4, 64, 8, 64, MainReducev2<int32_t, uint32_t> }));

TEST_P(Reducev2Testsuite, Reducev2TestCase)
{
    auto param = GetParam();
    uint8_t dstGm[param.dataSize3 * param.typeBytes];
    uint8_t src0Gm[param.dataSize1 * param.typeBytes];
    uint8_t src1Gm[param.dataSize2 * param.typeBytes];

    param.cal_func(dstGm, src0Gm, src1Gm, param.dataSize1, param.dataSize2, param.dataSize3);

    // for (int i = 0; i < param.dataSize2; i++) {
    //     EXPECT_EQ(dstGm[i], 0x00);
    // }
}
