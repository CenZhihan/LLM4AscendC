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
enum TestMode {
    LEVEL2,
    LEVEL0_BIT_MODE,
    LEVEL0_COUNT_MODE,
};

class TEST_CAST : public testing::Test {
protected:
    void SetUp() {
        g_coreType = AscendC::AIV_TYPE;
    }
    void TearDown() {
        g_coreType = AscendC::MIX_TYPE;
    }
};

template <typename DstType, typename SrcType>
void MainVecCastDemo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm, RoundMode roundMode,
    uint32_t dstDataSize, uint32_t srcDataSize, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<SrcType> inputGlobal;
    GlobalTensor<DstType> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ SrcType*>(srcGm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstType*>(dstGm), dstDataSize);

    LocalTensor<SrcType> inputLocal;
    LocalTensor<DstType> outputLocal;

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcDataSize * sizeof(SrcType));
    inputLocal = tbuf.Get<SrcType>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dstDataSize * sizeof(DstType));
    outputLocal = tbuf1.Get<DstType>();

    DataCopy(inputLocal, inputGlobal, srcDataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Cast(outputLocal, inputLocal, roundMode, dstDataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    pipe_barrier(PIPE_ALL);
}

void MainVecCastF162s4Demo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm, RoundMode roundMode,
    uint32_t dstDataSize, uint32_t srcDataSize, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<half> inputGlobal;
    GlobalTensor<int8_t> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(srcGm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(dstGm), dstDataSize);

    LocalTensor<half> inputLocal;
    LocalTensor<int8_t> outputLocal;

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcDataSize * sizeof(half));
    inputLocal = tbuf.Get<half>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dstDataSize * sizeof(int8_t));
    outputLocal = tbuf1.Get<int8_t>();

    DataCopy(inputLocal, inputGlobal, srcDataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    LocalTensor<int4b_t> outputLocalS4 = outputLocal.ReinterpretCast<int4b_t>();
    Cast(outputLocalS4, inputLocal, roundMode, srcDataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    pipe_barrier(PIPE_ALL);
}

#define VEC_CAST_TESTCASE(srcType, dstType, roundMode, testMode)                                      \
    TEST_F(TEST_CAST, Cast##srcType##2##dstType##roundMode##testMode##Case)                           \
    {                                                                                                     \
        uint32_t srcDataSize = 512;                                                                     \
        uint32_t dstDataSize = 512;                                                                     \
        uint8_t inputGm[srcDataSize * sizeof(srcType)];                                               \
        uint8_t outputGm[dstDataSize * sizeof(dstType)];                                              \
                                                                                                          \
        MainVecCastDemo<dstType, srcType>(outputGm, inputGm, RoundMode::roundMode, dstDataSize, \
            srcDataSize, testMode);                                                                    \
                                                                                                          \
        for (uint32_t i = 0; i < dstDataSize; i++) {                                                    \
            EXPECT_EQ(outputGm[i], 0x00);                                                                \
        }                                                                                                 \
    }

#define VEC_CAST_F162S4_TESTCASE(roundMode, testMode)                                                   \
    TEST_F(TEST_CAST, CastF162S4##roundMode##testMode##Case)                                            \
    {                                                                                                     \
        uint32_t srcDataSize = 128;                                                                     \
        uint32_t dstDataSize = 128 / INT4_TWO;                                                          \
        uint8_t inputGm[srcDataSize * sizeof(half)] = {0};                                             \
        uint8_t outputGm[dstDataSize] = {0};                                                           \
                                                                                                          \
        MainVecCastF162s4Demo(outputGm, inputGm, RoundMode::roundMode, dstDataSize,              \
            srcDataSize, testMode);                                                                    \
                                                                                                          \
        for (uint32_t i = 0; i < dstDataSize; i++) {                                                    \
            EXPECT_EQ(outputGm[i], 0x00);                                                                \
        }                                                                                                 \
    }

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002)
VEC_CAST_TESTCASE(half, int32_t, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(half, int32_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(half, int32_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(half, int32_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(half, int32_t, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(float, int32_t, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(float, int32_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(float, int32_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(float, int32_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(float, int32_t, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(int32_t, float, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(half, float, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(float, half, CAST_NONE, LEVEL2);
VEC_CAST_TESTCASE(float, half, CAST_ODD, LEVEL2);

VEC_CAST_TESTCASE(half, int8_t, CAST_NONE, LEVEL2);
VEC_CAST_TESTCASE(half, int8_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(half, int8_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(half, int8_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(half, int8_t, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(half, uint8_t, CAST_NONE, LEVEL2);
VEC_CAST_TESTCASE(half, uint8_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(half, uint8_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(half, uint8_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(half, uint8_t, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(int32_t, half, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(uint8_t, half, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(int8_t, half, CAST_NONE, LEVEL2);
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002)
VEC_CAST_TESTCASE(half, int16_t, CAST_RINT, LEVEL2);

VEC_CAST_TESTCASE(int16_t, half, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(int32_t, int16_t, CAST_NONE, LEVEL2);

VEC_CAST_F162S4_TESTCASE(CAST_NONE, LEVEL2);
#endif