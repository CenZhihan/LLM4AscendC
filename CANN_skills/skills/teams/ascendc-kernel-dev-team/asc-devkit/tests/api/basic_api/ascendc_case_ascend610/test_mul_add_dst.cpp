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

class TEST_MUL_ADD_DST : public testing::Test {
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

template <typename T, typename U>
void MainVecMulAddDstLevel2Demo(__gm__ uint8_t* __restrict__ dstGm,
    __gm__ uint8_t* __restrict__ src0Gm, __gm__ uint8_t* __restrict__ src1Gm, uint32_t dataSize)
{
    BinaryRepeatParams param;
    constexpr uint64_t maskCount = 256 / sizeof(T);
    constexpr uint8_t repeatTimes = sizeof(T);
    TPipe tpipe;
    GlobalTensor<U> input0Global;
    GlobalTensor<U> input1Global;
    GlobalTensor<T> inpoutputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(src0Gm), dataSize);
    input1Global.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(src1Gm), dataSize);
    inpoutputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(U));
    LocalTensor<U> input0Local = tbuf.Get<U>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(U));
    LocalTensor<U> input1Local = tbuf1.Get<U>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    MulAddDst(outputLocal, input0Local, input1Local, dataSize);
    if (sizeof(T) == sizeof(U)) {
        uint64_t mask[2] = {0xff, 0};
        MulAddDst(outputLocal, input0Local, input1Local, maskCount, repeatTimes, param);
        MulAddDst(outputLocal, input0Local, input1Local, mask, sizeof(T), param);
    } else {
        param.src0RepStride = 4;
        param.src1RepStride = 4;
        uint64_t mask[2] = {0xff, 0};
        MulAddDst(outputLocal, input0Local, input1Local, maskCount, repeatTimes, param);
        MulAddDst(outputLocal, input0Local, input1Local, mask, sizeof(T), param);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(inpoutputGlobal, outputLocal, dataSize);

    pipe_barrier(PIPE_ALL);
}

#define VEC_MUL_ADD_DST_LEVEL2_TESTCASE(dataType, srcType)                                              \
    TEST_F(TEST_MUL_ADD_DST, MulAddDst##dataType##srcType##Case)                                           \
    {                                                                                                           \
        uint32_t dataSize = 256;                                                                               \
        uint32_t selMaskSize = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));                         \
        uint8_t input0Gm[dataSize * sizeof(srcType)];                                                       \
        uint8_t input1Gm[dataSize * sizeof(srcType)];                                                       \
        uint8_t outputGm[dataSize * sizeof(dataType)];                                                       \
                                                                                                                \
        MainVecMulAddDstLevel2Demo<dataType, srcType>(outputGm, input0Gm, input1Gm, dataSize);     \
                                                                                                                \
        for (uint32_t i = 0; i < dataSize; i++) {                                                              \
            EXPECT_EQ(outputGm[i], 0x00);                                                                      \
        }                                                                                                       \
    }

VEC_MUL_ADD_DST_LEVEL2_TESTCASE(float, float);
VEC_MUL_ADD_DST_LEVEL2_TESTCASE(float, half);
VEC_MUL_ADD_DST_LEVEL2_TESTCASE(half, half);

