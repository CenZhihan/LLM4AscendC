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
struct SelfTensorDesc : public TensorTrait<T> {
    static constexpr int32_t tensorLen = 256;
};

template <typename T>
void MainVecBinaryScalarDemo(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm,
    __gm__ T scalarValue, __gm__ int32_t dataSize)
{
    AscendCUtils::SetOverflow(0);
    TPipe tpipe;
    GlobalTensor<SelfTensorDesc<T>> inputGlobal;
    GlobalTensor<SelfTensorDesc<T>> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> inputLocal = tbuf.Get<SelfTensorDesc<T>>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> outputLocal = tbuf1.Get<SelfTensorDesc<T>>();

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Mins(outputLocal, inputLocal, scalarValue, dataSize);
    Adds(outputLocal, outputLocal, scalarValue, dataSize);
    Muls(outputLocal, outputLocal, scalarValue, dataSize);
    Maxs(outputLocal, outputLocal, scalarValue, dataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
void MainVlreluDemo(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ T scalarValue,
    __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<SelfTensorDesc<T>> inputGlobal;
    GlobalTensor<SelfTensorDesc<T>> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> inputLocal = tbuf.Get<SelfTensorDesc<T>>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> outputLocal = tbuf1.Get<SelfTensorDesc<T>>();

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    LeakyRelu(outputLocal, inputLocal, scalarValue, dataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
void MainShiftDemo(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ T scalarValue,
    __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<SelfTensorDesc<T>> inputGlobal;
    GlobalTensor<SelfTensorDesc<T>> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> inputLocal = tbuf.Get<SelfTensorDesc<T>>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> outputLocal = tbuf1.Get<SelfTensorDesc<T>>();

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    ShiftRight(outputLocal, inputLocal, scalarValue, dataSize);
    ShiftLeft(outputLocal, inputLocal, scalarValue, dataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

class TEST_VEC_BINARY_SCALAR_DESC : public testing::Test {
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

TEST_F(TEST_VEC_BINARY_SCALAR_DESC, VecBinaryScalarDescFloatCase)
{
    int32_t dataSize = 208;
    int32_t scalar = 2;
    uint8_t srcGm[dataSize * sizeof(float)];
    uint8_t dstGm[dataSize * sizeof(float)];

    MainVecBinaryScalarDemo(srcGm, dstGm, static_cast<float>(scalar), dataSize);
    MainVlreluDemo(srcGm, dstGm, static_cast<float>(scalar), dataSize);
    for (int32_t i = 0; i < dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

TEST_F(TEST_VEC_BINARY_SCALAR_DESC, VecBinaryScalarDescHalfCase)
{
    int32_t dataSize = 208;
    int32_t scalar = 2;
    uint8_t srcGm[dataSize * sizeof(half)];
    uint8_t dstGm[dataSize * sizeof(half)];

    MainVecBinaryScalarDemo(srcGm, dstGm, static_cast<half>(scalar), dataSize);
    MainVlreluDemo(srcGm, dstGm, static_cast<half>(scalar), dataSize);
    for (int32_t i = 0; i < dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

TEST_F(TEST_VEC_BINARY_SCALAR_DESC, VecBinaryScalarDescInt16Case)
{
    int32_t dataSize = 208;
    int32_t scalar = 2;
    uint8_t srcGm[dataSize * sizeof(int16_t)];
    uint8_t dstGm[dataSize * sizeof(int16_t)];

    MainVecBinaryScalarDemo(srcGm, dstGm, static_cast<int16_t>(scalar), dataSize);
    MainShiftDemo(srcGm, dstGm, static_cast<int16_t>(scalar), dataSize);
    for (int32_t i = 0; i < dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

TEST_F(TEST_VEC_BINARY_SCALAR_DESC, VecBinaryScalarDescInt32Case)
{
    int32_t dataSize = 208;
    int32_t scalar = 2;
    uint8_t srcGm[dataSize * sizeof(int32_t)];
    uint8_t dstGm[dataSize * sizeof(int32_t)];

    MainVecBinaryScalarDemo(srcGm, dstGm, static_cast<int32_t>(scalar), dataSize);
    MainShiftDemo(srcGm, dstGm, static_cast<int32_t>(scalar), dataSize);
    for (int32_t i = 0; i < dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

TEST_F(TEST_VEC_BINARY_SCALAR_DESC, VecBinaryScalarDescUin16Case)
{
    int32_t dataSize = 208;
    int32_t scalar = 2;
    uint8_t srcGm[dataSize * sizeof(uint16_t)];
    uint8_t dstGm[dataSize * sizeof(uint16_t)];

    MainShiftDemo(srcGm, dstGm, static_cast<uint16_t>(scalar), dataSize);
    for (int32_t i = 0; i < dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

TEST_F(TEST_VEC_BINARY_SCALAR_DESC, VecBinaryScalarDescUin32Case)
{
    int32_t dataSize = 208;
    int32_t scalar = 2;
    uint8_t srcGm[dataSize * sizeof(uint32_t)];
    uint8_t dstGm[dataSize * sizeof(uint32_t)];

    MainShiftDemo(srcGm, dstGm, static_cast<uint32_t>(scalar), dataSize);
    for (int32_t i = 0; i < dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
