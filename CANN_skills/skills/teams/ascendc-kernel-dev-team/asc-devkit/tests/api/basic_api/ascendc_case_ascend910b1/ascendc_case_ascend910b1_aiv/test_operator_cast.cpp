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
#include "mockcpp/mockcpp.hpp"

using namespace std;
using namespace AscendC;
enum TestMode {
    LEVEL2,
    LEVEL0_BIT_MODE,
    LEVEL0_COUNT_MODE,
};

enum BinaryCastFunc {
    ADDRELUCAST,
    SUBRELUCAST,
};

class TestCast : public testing::Test {
protected:
    void SetUp() {
        g_coreType = AscendC::AIV_TYPE;
    }
    void TearDown() {
        g_coreType = AscendC::MIX_TYPE;
        GlobalMockObject::verify();
    }
};

class TestCastCompile : public testing::Test {
protected:
    void SetUp() {
        g_coreType = AscendC::AIV_TYPE;
    }
    void TearDown() {
        g_coreType = AscendC::MIX_TYPE;
        GlobalMockObject::verify();
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
    int s = sizeof(SrcType) > sizeof(DstType) ? sizeof(SrcType) : sizeof(DstType);
    int srcRepStride = 8;
    int dstRepStride = 8;
    if (sizeof(SrcType) > sizeof(DstType)) {
        srcRepStride = 8;
        dstRepStride = 8 / (sizeof(SrcType) / sizeof(DstType));
    }
    if (sizeof(SrcType) < sizeof(DstType)) {
        dstRepStride = 8;
        srcRepStride = 8 / (sizeof(DstType) / sizeof(SrcType));
    }
    if (testMode == LEVEL2) {
        if (sizeof(SrcType) < sizeof(DstType)) {
            Cast(outputLocal, outputLocal[dstDataSize / 2].template ReinterpretCast<SrcType>(), roundMode, dstDataSize);
        }
        if (std::is_same<SrcType, half>::value && std::is_same<DstType, int32_t>::value) {
            Cast(outputLocal, inputLocal, roundMode, 2048);
        }
        Cast(outputLocal, inputLocal, roundMode, dstDataSize);
    }
    if (testMode == LEVEL0_BIT_MODE) {
        uint64_t mask[2] = {0};
        uint8_t repeat = 0;
        if (s == 2) {
            mask[0] = 0xffffffffffffffff;
            mask[1] = 0xffffffffffffffff;
        } else if (s == 4) {
            mask[0] = 0xffffffffffffffff;
        } else if (s == 8) {
            mask[0] = 0x00000000ffffffff;
        }
        repeat = 2 * s;
        if (sizeof(SrcType) < sizeof(DstType)) {
            Cast(outputLocal, outputLocal[dstDataSize / 2].template ReinterpretCast<SrcType>(), roundMode,
                mask, repeat, { 1, 1, dstRepStride, srcRepStride });
        }
        if (std::is_same<SrcType, half>::value && std::is_same<DstType, int16_t>::value) {
            Cast(outputLocal, inputLocal, roundMode, mask, 4, { 1, 2, dstRepStride, 16 });
        }
        Cast(outputLocal, inputLocal, roundMode, mask, repeat, { 1, 1, dstRepStride, srcRepStride });
    }
    if (testMode == LEVEL0_COUNT_MODE) {
        uint64_t mask = 0;
        uint8_t repeat = 0;
        mask = 256 / s;
        repeat = 512 / mask;
        if (sizeof(SrcType) < sizeof(DstType)) {
            Cast(outputLocal, outputLocal[dstDataSize / 2].template ReinterpretCast<SrcType>(), roundMode,
                mask, repeat, { 1, 1, dstRepStride, srcRepStride });
        }
        if (std::is_same<SrcType, float>::value && std::is_same<DstType, int16_t>::value) {
            Cast(outputLocal, inputLocal, roundMode, 64, 8, { 1, 2, dstRepStride, 16 });
        }
        Cast(outputLocal, inputLocal, roundMode, mask, repeat, { 1, 1, dstRepStride, srcRepStride });
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    pipe_barrier(PIPE_ALL);
}

void MainVecCastS42f16Demo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm, RoundMode roundMode,
    uint32_t dstDataSize, uint32_t srcDataSize, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<int8_t> inputGlobal;
    GlobalTensor<half> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(srcGm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dstGm), dstDataSize);

    LocalTensor<int8_t> inputLocal;
    LocalTensor<half> outputLocal;

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcDataSize * sizeof(int8_t));
    inputLocal = tbuf.Get<int8_t>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dstDataSize * sizeof(half));
    outputLocal = tbuf1.Get<half>();

    DataCopy(inputLocal, inputGlobal, srcDataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    int srcRepStride = 2;
    int dstRepStride = 8;
    LocalTensor<int4b_t> inputLocalS4 = inputLocal.ReinterpretCast<int4b_t>();
    if (testMode == LEVEL2) {
        Cast(outputLocal, outputLocal[dstDataSize / 4 * 3].ReinterpretCast<int4b_t>(), roundMode, dstDataSize);
        Cast(outputLocal, inputLocalS4, roundMode, dstDataSize);
    }
    if (testMode == LEVEL0_BIT_MODE) {
        uint64_t mask[2] = {0xffffffffffffffff, 0xffffffffffffffff};
        uint8_t repeat = 1;
        Cast(outputLocal, outputLocal[dstDataSize / 4 * 3].ReinterpretCast<int4b_t>(), roundMode,
            mask, repeat, { 1, 1, dstRepStride, srcRepStride });
        Cast(outputLocal, inputLocalS4, roundMode, mask, repeat, { 1, 1, dstRepStride, srcRepStride });
    }
    if (testMode == LEVEL0_COUNT_MODE) {
        uint64_t mask = 128;
        uint8_t repeat = 2;
        Cast(outputLocal, outputLocal[dstDataSize / 4 * 3].ReinterpretCast<int4b_t>(), roundMode,
            mask, repeat, { 1, 1, dstRepStride, srcRepStride });
        Cast(outputLocal, inputLocalS4, roundMode, mask, repeat, { 1, 1, dstRepStride, srcRepStride });
    }
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
    int srcRepStride = 8;
    int dstRepStride = 2;
    LocalTensor<int4b_t> outputLocalS4 = outputLocal.ReinterpretCast<int4b_t>();
    if (testMode == LEVEL2) {
        Cast(outputLocalS4, inputLocal, roundMode, srcDataSize);
    }
    if (testMode == LEVEL0_BIT_MODE) {
        uint64_t mask[2] = {0xffffffffffffffff, 0xffffffffffffffff};
        uint8_t repeat = 1;
        Cast(outputLocalS4, inputLocal, roundMode, mask, repeat, { 1, 1, dstRepStride, srcRepStride });
    }
    if (testMode == LEVEL0_COUNT_MODE) {
        uint64_t mask = 128;
        uint8_t repeat = 2;
        Cast(outputLocalS4, inputLocal, roundMode, mask, repeat, { 1, 1, dstRepStride, srcRepStride });
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    pipe_barrier(PIPE_ALL);
}

int32_t RaiseStub(int32_t input)
{
    return 0;
}

#define VEC_CAST_TESTCASE_2(srcType, dstType, roundMode, testMode, errorTimes)                       \
    TEST_F(TestCast, Cast##srcType##2##dstType##roundMode##testMode##case)                           \
    {                                                                                                     \
        uint32_t srcDataSize = 512;                                                                     \
        uint32_t dstDataSize = 512;                                                                     \
        uint8_t inputGm[srcDataSize * sizeof(srcType)] = {0};                                         \
        uint8_t outputGm[dstDataSize * sizeof(dstType)] = {0};                                        \
        MOCKER(raise, int32_t (*)(int32_t)).times(1).will(invoke(RaiseStub));                             \
        MainVecCastDemo<dstType, srcType>(outputGm, inputGm, RoundMode::roundMode, dstDataSize, \
            srcDataSize, testMode);                                                                    \
                                                                                                          \
        for (uint32_t i = 0; i < dstDataSize; i++) {                                                    \
            EXPECT_EQ(outputGm[i], 0x00);                                                                \
        }                                                                                                 \
    }

#define VEC_CAST_TESTCASE(srcType, dstType, roundMode, testMode)                                      \
    TEST_F(TestCast, Cast##srcType##2##dstType##roundMode##testMode##case)                           \
    {                                                                                                     \
        uint32_t srcDataSize = 512;                                                                     \
        uint32_t dstDataSize = 512;                                                                     \
        uint8_t inputGm[srcDataSize * sizeof(srcType)] = {0};                                         \
        uint8_t outputGm[dstDataSize * sizeof(dstType)] = {0};                                        \
                                                                                                          \
        MainVecCastDemo<dstType, srcType>(outputGm, inputGm, RoundMode::roundMode, dstDataSize, \
            srcDataSize, testMode);                                                                    \
                                                                                                          \
        for (uint32_t i = 0; i < dstDataSize; i++) {                                                    \
            EXPECT_EQ(outputGm[i], 0x00);                                                                \
        }                                                                                                 \
    }

#define VEC_CAST_S42F16_TESTCASE(roundMode, testMode)                                                   \
    TEST_F(TestCast, CastS42F16##roundMode##testMode##case)                                            \
    {                                                                                                     \
        uint32_t srcDataSize = 512 / INT4_TWO;                                                          \
        uint32_t dstDataSize = 512;                                                                     \
        uint8_t inputGm[srcDataSize] = {0};                                                            \
        uint8_t outputGm[dstDataSize * sizeof(half)] = {0};                                            \
                                                                                                          \
        MainVecCastS42f16Demo(outputGm, inputGm, RoundMode::roundMode, dstDataSize,              \
            srcDataSize, testMode);                                                                    \
                                                                                                          \
        for (uint32_t i = 0; i < dstDataSize; i++) {                                                    \
            EXPECT_EQ(outputGm[i], 0x00);                                                                \
        }                                                                                                 \
    }

#define VEC_CAST_F162S4_TESTCASE(roundMode, testMode)                                                   \
    TEST_F(TestCast, CastF162S4##roundMode##testMode##case)                                            \
    {                                                                                                     \
        uint32_t srcDataSize = 512;                                                                     \
        uint32_t dstDataSize = 512 / INT4_TWO;                                                          \
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

VEC_CAST_TESTCASE(int32_t, half, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(int8_t, half, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(uint8_t, half, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(int32_t, float, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(int32_t, float, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(int32_t, float, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(int32_t, float, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(int32_t, float, CAST_TRUNC, LEVEL2);
VEC_CAST_TESTCASE(int32_t, float, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(half, float, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE_2(half, int32_t, CAST_RINT, LEVEL2, 1);
VEC_CAST_TESTCASE_2(half, int32_t, CAST_FLOOR, LEVEL2, 1);
VEC_CAST_TESTCASE_2(half, int32_t, CAST_CEIL, LEVEL2, 1);
VEC_CAST_TESTCASE_2(half, int32_t, CAST_ROUND, LEVEL2, 1);
VEC_CAST_TESTCASE_2(half, int32_t, CAST_TRUNC, LEVEL2, 1);

VEC_CAST_TESTCASE(half, int8_t, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(half, int8_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(half, int8_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(half, int8_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(half, int8_t, CAST_TRUNC, LEVEL2);
VEC_CAST_TESTCASE(half, int8_t, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(half, uint8_t, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(half, uint8_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(half, uint8_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(half, uint8_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(half, uint8_t, CAST_TRUNC, LEVEL2);
VEC_CAST_TESTCASE(half, uint8_t, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(float, half, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(float, half, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(float, half, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(float, half, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(float, half, CAST_TRUNC, LEVEL2);
VEC_CAST_TESTCASE(float, half, CAST_ODD, LEVEL2);
VEC_CAST_TESTCASE(float, half, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(float, int32_t, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(float, int32_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(float, int32_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(float, int32_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(float, int32_t, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(half, int16_t, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(half, int16_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(half, int16_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(half, int16_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(half, int16_t, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(int16_t, half, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(int16_t, half, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(int16_t, half, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(int16_t, half, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(int16_t, half, CAST_TRUNC, LEVEL2);
VEC_CAST_TESTCASE(int16_t, half, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(float, float, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(float, float, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(float, float, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(float, float, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(float, float, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(float, bfloat16_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(float, bfloat16_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(float, bfloat16_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(float, bfloat16_t, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(float, bfloat16_t, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(float, int64_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(float, int64_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(float, int64_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(float, int64_t, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(float, int64_t, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(bfloat16_t, float, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(float, int16_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(float, int16_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(float, int16_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(float, int16_t, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(float, int16_t, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(int16_t, float, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(int32_t, int16_t, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(int32_t, int64_t, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(int64_t, float, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(int64_t, float, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(int64_t, float, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(int64_t, float, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(int64_t, float, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(int64_t, int32_t, CAST_NONE, LEVEL2);

VEC_CAST_F162S4_TESTCASE(CAST_RINT, LEVEL2);
VEC_CAST_F162S4_TESTCASE(CAST_FLOOR, LEVEL2);
VEC_CAST_F162S4_TESTCASE(CAST_CEIL, LEVEL2);
VEC_CAST_F162S4_TESTCASE(CAST_ROUND, LEVEL2);
VEC_CAST_F162S4_TESTCASE(CAST_TRUNC, LEVEL2);
VEC_CAST_F162S4_TESTCASE(CAST_NONE, LEVEL2);

VEC_CAST_S42F16_TESTCASE(CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(int32_t, half, CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(int8_t, half, CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(uint8_t, half, CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(int32_t, float, CAST_RINT, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(int32_t, float, CAST_FLOOR, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(int32_t, float, CAST_CEIL, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(int32_t, float, CAST_ROUND, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(int32_t, float, CAST_TRUNC, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(int32_t, float, CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(half, float, CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(half, int32_t, CAST_RINT, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(half, int32_t, CAST_FLOOR, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(half, int32_t, CAST_CEIL, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(half, int32_t, CAST_ROUND, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(half, int32_t, CAST_TRUNC, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(half, int8_t, CAST_RINT, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(half, int8_t, CAST_FLOOR, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(half, int8_t, CAST_CEIL, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(half, int8_t, CAST_ROUND, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(half, int8_t, CAST_TRUNC, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(half, int8_t, CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(half, uint8_t, CAST_RINT, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(half, uint8_t, CAST_FLOOR, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(half, uint8_t, CAST_CEIL, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(half, uint8_t, CAST_ROUND, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(half, uint8_t, CAST_TRUNC, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(half, uint8_t, CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(float, half, CAST_RINT, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, half, CAST_FLOOR, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, half, CAST_CEIL, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, half, CAST_ROUND, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, half, CAST_TRUNC, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, half, CAST_ODD, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, half, CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(float, int32_t, CAST_RINT, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, int32_t, CAST_FLOOR, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, int32_t, CAST_CEIL, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, int32_t, CAST_ROUND, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, int32_t, CAST_TRUNC, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE_2(half, int16_t, CAST_RINT, LEVEL0_BIT_MODE, 1);
VEC_CAST_TESTCASE_2(half, int16_t, CAST_ROUND, LEVEL0_BIT_MODE, 1);
VEC_CAST_TESTCASE_2(half, int16_t, CAST_CEIL, LEVEL0_BIT_MODE, 1);
VEC_CAST_TESTCASE_2(half, int16_t, CAST_FLOOR, LEVEL0_BIT_MODE, 1);
VEC_CAST_TESTCASE_2(half, int16_t, CAST_TRUNC, LEVEL0_BIT_MODE, 1);

VEC_CAST_TESTCASE(int16_t, half, CAST_ROUND, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(int16_t, half, CAST_CEIL, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(int16_t, half, CAST_FLOOR, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(int16_t, half, CAST_RINT, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(int16_t, half, CAST_TRUNC, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(int16_t, half, CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(float, float, CAST_ROUND, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, float, CAST_CEIL, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, float, CAST_FLOOR, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, float, CAST_RINT, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, float, CAST_TRUNC, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(float, bfloat16_t, CAST_ROUND, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, bfloat16_t, CAST_CEIL, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, bfloat16_t, CAST_FLOOR, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, bfloat16_t, CAST_RINT, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, bfloat16_t, CAST_TRUNC, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(float, int64_t, CAST_ROUND, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, int64_t, CAST_CEIL, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, int64_t, CAST_FLOOR, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, int64_t, CAST_RINT, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, int64_t, CAST_TRUNC, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(bfloat16_t, float, CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_ROUND, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_CEIL, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_FLOOR, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_RINT, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_TRUNC, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(float, int16_t, CAST_ROUND, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, int16_t, CAST_CEIL, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, int16_t, CAST_FLOOR, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, int16_t, CAST_RINT, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(float, int16_t, CAST_TRUNC, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(int16_t, float, CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(int32_t, int16_t, CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(int32_t, int64_t, CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(int64_t, float, CAST_ROUND, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(int64_t, float, CAST_CEIL, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(int64_t, float, CAST_FLOOR, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(int64_t, float, CAST_RINT, LEVEL0_BIT_MODE);
VEC_CAST_TESTCASE(int64_t, float, CAST_TRUNC, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(int64_t, int32_t, CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_F162S4_TESTCASE(CAST_RINT, LEVEL0_BIT_MODE);
VEC_CAST_F162S4_TESTCASE(CAST_FLOOR, LEVEL0_BIT_MODE);
VEC_CAST_F162S4_TESTCASE(CAST_CEIL, LEVEL0_BIT_MODE);
VEC_CAST_F162S4_TESTCASE(CAST_ROUND, LEVEL0_BIT_MODE);
VEC_CAST_F162S4_TESTCASE(CAST_TRUNC, LEVEL0_BIT_MODE);
VEC_CAST_F162S4_TESTCASE(CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_S42F16_TESTCASE(CAST_NONE, LEVEL0_BIT_MODE);

VEC_CAST_TESTCASE(int32_t, half, CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(int8_t, half, CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(uint8_t, half, CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(int32_t, float, CAST_RINT, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(int32_t, float, CAST_FLOOR, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(int32_t, float, CAST_CEIL, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(int32_t, float, CAST_ROUND, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(int32_t, float, CAST_TRUNC, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(int32_t, float, CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(half, float, CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(half, int32_t, CAST_RINT, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, int32_t, CAST_FLOOR, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, int32_t, CAST_CEIL, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, int32_t, CAST_ROUND, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, int32_t, CAST_TRUNC, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(half, int8_t, CAST_RINT, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, int8_t, CAST_FLOOR, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, int8_t, CAST_CEIL, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, int8_t, CAST_ROUND, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, int8_t, CAST_TRUNC, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, int8_t, CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(half, uint8_t, CAST_RINT, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, uint8_t, CAST_FLOOR, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, uint8_t, CAST_CEIL, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, uint8_t, CAST_ROUND, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, uint8_t, CAST_TRUNC, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, uint8_t, CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(float, half, CAST_RINT, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, half, CAST_FLOOR, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, half, CAST_CEIL, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, half, CAST_ROUND, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, half, CAST_TRUNC, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, half, CAST_ODD, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, half, CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(float, int32_t, CAST_RINT, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, int32_t, CAST_FLOOR, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, int32_t, CAST_CEIL, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, int32_t, CAST_ROUND, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, int32_t, CAST_TRUNC, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(half, int16_t, CAST_RINT, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, int16_t, CAST_ROUND, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, int16_t, CAST_CEIL, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, int16_t, CAST_FLOOR, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(half, int16_t, CAST_TRUNC, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(int16_t, half, CAST_ROUND, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(int16_t, half, CAST_CEIL, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(int16_t, half, CAST_FLOOR, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(int16_t, half, CAST_RINT, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(int16_t, half, CAST_TRUNC, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(int16_t, half, CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(float, float, CAST_ROUND, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, float, CAST_CEIL, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, float, CAST_FLOOR, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, float, CAST_RINT, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, float, CAST_TRUNC, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(float, bfloat16_t, CAST_ROUND, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, bfloat16_t, CAST_CEIL, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, bfloat16_t, CAST_FLOOR, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, bfloat16_t, CAST_RINT, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, bfloat16_t, CAST_TRUNC, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(float, int64_t, CAST_ROUND, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, int64_t, CAST_CEIL, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, int64_t, CAST_FLOOR, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, int64_t, CAST_RINT, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(float, int64_t, CAST_TRUNC, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(bfloat16_t, float, CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_ROUND, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_CEIL, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_FLOOR, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_RINT, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(bfloat16_t, int32_t, CAST_TRUNC, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE_2(float, int16_t, CAST_ROUND, LEVEL0_COUNT_MODE, 1);
VEC_CAST_TESTCASE_2(float, int16_t, CAST_CEIL, LEVEL0_COUNT_MODE, 1);
VEC_CAST_TESTCASE_2(float, int16_t, CAST_FLOOR, LEVEL0_COUNT_MODE, 1);
VEC_CAST_TESTCASE_2(float, int16_t, CAST_RINT, LEVEL0_COUNT_MODE, 1);
VEC_CAST_TESTCASE_2(float, int16_t, CAST_TRUNC, LEVEL0_COUNT_MODE, 1);

VEC_CAST_TESTCASE(int16_t, float, CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(int32_t, int16_t, CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(int32_t, int64_t, CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(int64_t, float, CAST_ROUND, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(int64_t, float, CAST_CEIL, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(int64_t, float, CAST_FLOOR, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(int64_t, float, CAST_RINT, LEVEL0_COUNT_MODE);
VEC_CAST_TESTCASE(int64_t, float, CAST_TRUNC, LEVEL0_COUNT_MODE);

VEC_CAST_TESTCASE(int64_t, int32_t, CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_F162S4_TESTCASE(CAST_RINT, LEVEL0_COUNT_MODE);
VEC_CAST_F162S4_TESTCASE(CAST_FLOOR, LEVEL0_COUNT_MODE);
VEC_CAST_F162S4_TESTCASE(CAST_CEIL, LEVEL0_COUNT_MODE);
VEC_CAST_F162S4_TESTCASE(CAST_ROUND, LEVEL0_COUNT_MODE);
VEC_CAST_F162S4_TESTCASE(CAST_TRUNC, LEVEL0_COUNT_MODE);
VEC_CAST_F162S4_TESTCASE(CAST_NONE, LEVEL0_COUNT_MODE);

VEC_CAST_S42F16_TESTCASE(CAST_NONE, LEVEL0_COUNT_MODE);

template <typename DstType, typename SrcType>
void MainVecBinaryCastDemo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ src1Gm, uint32_t dstDataSize, uint32_t srcDataSize, BinaryCastFunc funcName,
    TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<SrcType> input0Global;
    GlobalTensor<SrcType> input1Global;
    GlobalTensor<DstType> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ SrcType*>(src0Gm), srcDataSize);
    input1Global.SetGlobalBuffer(reinterpret_cast<__gm__ SrcType*>(src1Gm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstType*>(dstGm), dstDataSize);

    LocalTensor<SrcType> input0Local;
    LocalTensor<SrcType> input1Local;
    LocalTensor<DstType> outputLocal;

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcDataSize * sizeof(SrcType));
    input0Local = tbuf.Get<SrcType>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, srcDataSize * sizeof(SrcType));
    input1Local = tbuf1.Get<SrcType>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dstDataSize * sizeof(DstType));
    outputLocal = tbuf2.Get<DstType>();

    DataCopy(input0Local, input0Global, srcDataSize);
    DataCopy(input1Local, input1Global, srcDataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    int s = sizeof(SrcType) > sizeof(DstType) ? sizeof(SrcType) : sizeof(DstType);
    int srcRepStride = 8;
    int dstRepStride = 8;
    if (sizeof(SrcType) > sizeof(DstType)) {
        srcRepStride = 8;
        dstRepStride = 8 / (sizeof(SrcType) / sizeof(DstType));
    }
    if (sizeof(SrcType) < sizeof(DstType)) {
        dstRepStride = 8;
        srcRepStride = 8 / (sizeof(DstType) / sizeof(SrcType));
    }

    if (testMode == LEVEL2) {
        switch (funcName) {
            case BinaryCastFunc::ADDRELUCAST:
                if (std::is_same<DstType, half>::value) {
                    AddReluCast(outputLocal, input0Local, input1Local, 1024);
                }
                AddReluCast(outputLocal, input0Local, input1Local, dstDataSize);
                break;
            case BinaryCastFunc::SUBRELUCAST:
                if (std::is_same<DstType, half>::value) {
                    SubReluCast(outputLocal, input0Local, input1Local, 1024);
                }
                SubReluCast(outputLocal, input0Local, input1Local, dstDataSize);
                break;
            default:
                ASCENDC_ASSERT(
                    (0), { KERNEL_LOG(KERNEL_ERROR, "invalid funcName %d", static_cast<int32_t>(funcName)); });
                break;
        }
    }
    if (testMode == LEVEL0_BIT_MODE) {
        uint64_t mask[2] = {0};
        uint8_t repeat = 0;
        if (s == 2) {
            mask[0] = 0xffffffffffffffff;
            mask[1] = 0xffffffffffffffff;
        } else if (s == 4) {
            mask[0] = 0xffffffffffffffff;
        } else if (s == 8) {
            mask[0] = 0x00000000ffffffff;
        }
        repeat = 2 * s;
        switch (funcName) {
            case BinaryCastFunc::ADDRELUCAST:
                if (std::is_same<DstType, half>::value) {
                    AddReluCast(outputLocal, input0Local, input1Local, mask, 128,
                        { 1, 2, 1, dstRepStride, 16, srcRepStride });
                }
                AddReluCast(outputLocal, input0Local, input1Local, mask, repeat,
                    { 1, 1, 1, dstRepStride, srcRepStride, srcRepStride });
                break;
            case BinaryCastFunc::SUBRELUCAST:
                if (std::is_same<DstType, half>::value) {
                    SubReluCast(outputLocal, input0Local, input1Local, mask, 128,
                        { 1, 2, 1, dstRepStride, 16, srcRepStride });
                }
                SubReluCast(outputLocal, input0Local, input1Local, mask, repeat,
                    { 1, 1, 1, dstRepStride, srcRepStride, srcRepStride });
                break;
            default:
                ASCENDC_ASSERT(
                    (0), { KERNEL_LOG(KERNEL_ERROR, "invalid funcName %d", static_cast<int32_t>(funcName)); });
                break;
        }
    }
    if (testMode == LEVEL0_COUNT_MODE) {
        uint64_t mask = 0;
        uint8_t repeat = 0;
        mask = 256 / s;
        repeat = 512 / mask;
        switch (funcName) {
            case BinaryCastFunc::ADDRELUCAST:
                if (std::is_same<SrcType, half>::value) {
                    AddReluCast(outputLocal, input0Local, input1Local, mask, 16,
                        { 1, 1, 1, dstRepStride, srcRepStride, srcRepStride });
                }
                AddReluCast(outputLocal, input0Local, input1Local, mask, repeat,
                    { 1, 1, 1, dstRepStride, srcRepStride, srcRepStride });
                break;
            case BinaryCastFunc::SUBRELUCAST:
                if (std::is_same<SrcType, half>::value) {
                    SubReluCast(outputLocal, input0Local, input1Local, mask, 16,
                        { 1, 1, 1, dstRepStride, srcRepStride, srcRepStride });
                }
                SubReluCast(outputLocal, input0Local, input1Local, mask, repeat,
                    { 1, 1, 1, dstRepStride, srcRepStride, srcRepStride });
                break;
            default:
                ASCENDC_ASSERT(
                    (0), { KERNEL_LOG(KERNEL_ERROR, "invalid funcName %d", static_cast<int32_t>(funcName)); });
                break;
        }
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    pipe_barrier(PIPE_ALL);

    int32_t tmp = g_coreType;
    g_coreType = AscendC::AIC_TYPE;
    LocalTensor<float> src0;
    LocalTensor<float> src1;
    LocalTensor<half> dst;
    uint64_t mask = 64;
    uint64_t bitMask[2] = {0xffffffffffffffff, 0};
    AddReluCast(dst, src0, src1, dstDataSize);
    SubReluCast(dst, src0, src1, dstDataSize);
    AddReluCast(dst, src0, src1, mask, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE / HALF_FACTOR, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    SubReluCast(dst, src0, src1, mask, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE / HALF_FACTOR, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    AddReluCast(dst, src0, src1, bitMask, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE / HALF_FACTOR, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    SubReluCast(dst, src0, src1, bitMask, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE / HALF_FACTOR, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    g_coreType = tmp;

    pipe_barrier(PIPE_ALL);
}

#define VEC_BINARY_CAST_TESTCASE_2(srcType, dstType, funcName, testMode, errorTimes)                 \
    TEST_F(TestCast, Test##funcName##srcType##2##dstType##testMode##case)                            \
    {                                                                                                     \
        uint32_t srcDataSize = 512;                                                                     \
        uint32_t dstDataSize = 512;                                                                     \
        uint8_t input0Gm[srcDataSize * sizeof(srcType)] = {0};                                        \
        uint8_t input1Gm[srcDataSize * sizeof(srcType)] = {0};                                        \
        uint8_t outputGm[dstDataSize * sizeof(dstType)] = {0};                                        \
        MOCKER(raise, int32_t (*)(int32_t)).times(1).will(invoke(RaiseStub));                             \
        MainVecBinaryCastDemo<dstType, srcType>(outputGm, input0Gm, input1Gm, dstDataSize,     \
            srcDataSize, funcName, testMode);                                                         \
                                                                                                          \
        for (uint32_t i = 0; i < dstDataSize; i++) {                                                    \
            EXPECT_EQ(outputGm[i], 0x00);                                                                \
        }                                                                                                 \
    }

#define VEC_BINARY_CAST_TESTCASE(srcType, dstType, funcName, testMode)                                \
    TEST_F(TestCast, Test##funcName##srcType##2##dstType##testMode##case)                            \
    {                                                                                                     \
        uint32_t srcDataSize = 512;                                                                     \
        uint32_t dstDataSize = 512;                                                                     \
        uint8_t input0Gm[srcDataSize * sizeof(srcType)] = {0};                                        \
        uint8_t input1Gm[srcDataSize * sizeof(srcType)] = {0};                                        \
        uint8_t outputGm[dstDataSize * sizeof(dstType)] = {0};                                        \
                                                                                                          \
        MainVecBinaryCastDemo<dstType, srcType>(outputGm, input0Gm, input1Gm, dstDataSize,     \
            srcDataSize, funcName, testMode);                                                         \
                                                                                                          \
        for (uint32_t i = 0; i < dstDataSize; i++) {                                                    \
            EXPECT_EQ(outputGm[i], 0x00);                                                                \
        }                                                                                                 \
    }

VEC_BINARY_CAST_TESTCASE(half, int8_t, ADDRELUCAST, LEVEL2);
VEC_BINARY_CAST_TESTCASE_2(float, half, ADDRELUCAST, LEVEL2, 1);
VEC_BINARY_CAST_TESTCASE(int16_t, int8_t, ADDRELUCAST, LEVEL2);

VEC_BINARY_CAST_TESTCASE(half, int8_t, SUBRELUCAST, LEVEL2);
VEC_BINARY_CAST_TESTCASE_2(float, half, SUBRELUCAST, LEVEL2, 1);
VEC_BINARY_CAST_TESTCASE(int16_t, int8_t, SUBRELUCAST, LEVEL2);

VEC_BINARY_CAST_TESTCASE(half, int8_t, ADDRELUCAST, LEVEL0_BIT_MODE);
VEC_BINARY_CAST_TESTCASE_2(float, half, ADDRELUCAST, LEVEL0_BIT_MODE, 1);
VEC_BINARY_CAST_TESTCASE(int16_t, int8_t, ADDRELUCAST, LEVEL0_BIT_MODE);

VEC_BINARY_CAST_TESTCASE(half, int8_t, SUBRELUCAST, LEVEL0_BIT_MODE);
VEC_BINARY_CAST_TESTCASE_2(float, half, SUBRELUCAST, LEVEL0_BIT_MODE, 1);
VEC_BINARY_CAST_TESTCASE(int16_t, int8_t, SUBRELUCAST, LEVEL0_BIT_MODE);

VEC_BINARY_CAST_TESTCASE_2(half, int8_t, ADDRELUCAST, LEVEL0_COUNT_MODE, 1);
VEC_BINARY_CAST_TESTCASE(float, half, ADDRELUCAST, LEVEL0_COUNT_MODE);
VEC_BINARY_CAST_TESTCASE(int16_t, int8_t, ADDRELUCAST, LEVEL0_COUNT_MODE);

VEC_BINARY_CAST_TESTCASE_2(half, int8_t, SUBRELUCAST, LEVEL0_COUNT_MODE, 1);
VEC_BINARY_CAST_TESTCASE(float, half, SUBRELUCAST, LEVEL0_COUNT_MODE);
VEC_BINARY_CAST_TESTCASE(int16_t, int8_t, SUBRELUCAST, LEVEL0_COUNT_MODE);

template <typename DstType, typename SrcType>
void MainVecCastDemoDeath(RoundMode roundMode, uint32_t dstDataSize, uint32_t srcDataSize, TestMode testMode)
{
    TPipe tpipe;
    LocalTensor<SrcType> inputLocal;
    LocalTensor<DstType> outputLocal;

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcDataSize * sizeof(SrcType));
    inputLocal = tbuf.Get<SrcType>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dstDataSize * sizeof(DstType));
    outputLocal = tbuf1.Get<DstType>();
    Cast(outputLocal, inputLocal, roundMode, dstDataSize);
}

#define VEC_CAST_TESTCASE(srcType, dstType, roundMode, testMode)                                      \
    TEST_F(TestCast, Cast##srcType##2##dstType##roundMode##testMode##case)                           \
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

#define VEC_CAST_FAILED_TESTCASE(srcType, dstType, roundMode, testMode)                                           \
    TEST_F(TestCastCompile, Cast##srcType##2##dstType##roundMode##testMode##FailedCase)                         \
    {                                                                                                                 \
        uint32_t srcDataSize = 512;                                                                                 \
        uint32_t dstDataSize = 512;                                                                                 \
        MOCKER(raise, int32_t (*)(int32_t)).times(1).will(invoke(RaiseStub));                                         \
        MainVecCastDemoDeath<dstType, srcType>(RoundMode::roundMode, dstDataSize, srcDataSize, testMode); \
    }

VEC_CAST_FAILED_TESTCASE(int32_t, float, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(half, int32_t, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(half, int32_t, CAST_NONE, LEVEL2);
VEC_CAST_FAILED_TESTCASE(half, int8_t, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(half, uint8_t, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, int32_t, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, int32_t, CAST_NONE, LEVEL2);
VEC_CAST_FAILED_TESTCASE(half, int16_t, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(half, int16_t, CAST_NONE, LEVEL2);
VEC_CAST_FAILED_TESTCASE(int16_t, half, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, float, CAST_NONE, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, float, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, bfloat16_t, CAST_NONE, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, bfloat16_t, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, int64_t, CAST_NONE, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, int64_t, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(bfloat16_t, int32_t, CAST_NONE, LEVEL2);
VEC_CAST_FAILED_TESTCASE(bfloat16_t, int32_t, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, int16_t, CAST_NONE, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, int16_t, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(int64_t, float, CAST_NONE, LEVEL2);
VEC_CAST_FAILED_TESTCASE(int64_t, float, CAST_ODD, LEVEL2);