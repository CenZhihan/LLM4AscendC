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

template <typename T>
void MainVecBinaryScalarErrorDemo(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm,
    __gm__ T scalarValue, __gm__ int32_t dataSize)
{
    AscendCUtils::SetOverflow(0);
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    UnaryRepeatParams unaryParams;
    uint64_t mask = 256 / sizeof(T);
    uint64_t maskbit[2] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
    int32_t dataSizeTmp = 512;
    Mins(outputLocal, inputLocal, scalarValue, dataSizeTmp);
    Mins(outputLocal, inputLocal, scalarValue, mask, dataSizeTmp / mask, unaryParams);
    Mins(outputLocal, inputLocal, scalarValue, maskbit, dataSizeTmp / mask, unaryParams);
    Adds(outputLocal, outputLocal, scalarValue, dataSizeTmp);
    Adds(outputLocal, inputLocal, scalarValue, mask, dataSizeTmp / mask, unaryParams);
    Adds(outputLocal, inputLocal, scalarValue, maskbit, dataSizeTmp / mask, unaryParams);
    Muls(outputLocal, outputLocal, scalarValue, dataSizeTmp);
    Muls(outputLocal, inputLocal, scalarValue, mask, dataSizeTmp / mask, unaryParams);
    Muls(outputLocal, inputLocal, scalarValue, maskbit, dataSizeTmp / mask, unaryParams);
    Maxs(outputLocal, outputLocal, scalarValue, dataSizeTmp);
    Maxs(outputLocal, inputLocal, scalarValue, mask, dataSizeTmp / mask, unaryParams);
    Maxs(outputLocal, inputLocal, scalarValue, maskbit, dataSizeTmp / mask, unaryParams);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
void MainVlreluErrorDemo(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ T scalarValue,
    __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    UnaryRepeatParams unaryParams;
    uint64_t mask = 256 / sizeof(T);
    uint64_t maskbit[2] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
    int32_t dataSizeTmp = 512;
    LeakyRelu(outputLocal, inputLocal, scalarValue, dataSizeTmp);
    LeakyRelu(outputLocal, inputLocal, scalarValue, mask, dataSizeTmp / mask, unaryParams);
    LeakyRelu(outputLocal, inputLocal, scalarValue, maskbit, dataSizeTmp / mask, unaryParams);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
void MainShiftErrorDemo(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm,
    __gm__ T scalarValue, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    UnaryRepeatParams unaryParams;
    uint64_t mask = 256 / sizeof(T);
    uint64_t maskbit[2] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
    int32_t dataSizeTmp = 512;
    ShiftRight(outputLocal, inputLocal, scalarValue, dataSizeTmp);
    ShiftRight(outputLocal, inputLocal, scalarValue, mask, dataSizeTmp / mask, unaryParams);
    ShiftRight(outputLocal, inputLocal, scalarValue, maskbit, dataSizeTmp / mask, unaryParams);
    ShiftLeft(outputLocal, inputLocal, scalarValue, dataSizeTmp);
    ShiftLeft(outputLocal, inputLocal, scalarValue, mask, dataSizeTmp / mask, unaryParams);
    ShiftLeft(outputLocal, inputLocal, scalarValue, maskbit, dataSizeTmp / mask, unaryParams);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

class TEST_VEC_BINARY_SCALAR_DESC_ERROR : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(2);
    }

    void TearDown()
    {
        AscendC::SetGCoreType(0);
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_VEC_BINARY_SCALAR_DESC_ERROR, VecBinaryScalarErrorHalfCase)
{
    int32_t dataSize = 208;
    int32_t scalar = 2;
    uint8_t srcGm[dataSize * sizeof(half)];
    uint8_t dstGm[dataSize * sizeof(half)];
    MOCKER(raise, int(*)(int)).times(15).will(returnValue(0));
    MainVecBinaryScalarErrorDemo(srcGm, dstGm, static_cast<half>(scalar), dataSize);
    MainVlreluErrorDemo(srcGm, dstGm, static_cast<half>(scalar), dataSize);
    for (int32_t i = 0; i < dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

TEST_F(TEST_VEC_BINARY_SCALAR_DESC_ERROR, VecBinaryScalarErrorUin16Case)
{
    int32_t dataSize = 208;
    int32_t scalar = 2;
    uint8_t srcGm[dataSize * sizeof(uint16_t)];
    uint8_t dstGm[dataSize * sizeof(uint16_t)];
    MOCKER(raise, int(*)(int)).times(6).will(returnValue(0));
    MainShiftErrorDemo(srcGm, dstGm, static_cast<uint16_t>(scalar), dataSize);
    for (int32_t i = 0; i < dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

