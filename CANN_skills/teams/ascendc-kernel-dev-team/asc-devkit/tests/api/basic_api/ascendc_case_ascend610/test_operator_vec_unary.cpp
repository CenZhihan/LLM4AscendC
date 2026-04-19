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

template <typename T, bool USE_CAST_DEQ>
void MainVecUnary01(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
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
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Exp(outputLocal, inputLocal, dataSize);
    Ln(outputLocal, inputLocal, dataSize);
    Abs(outputLocal, inputLocal, dataSize);
    Reciprocal(outputLocal, inputLocal, dataSize);
    Rsqrt(outputLocal, inputLocal, dataSize);
    Sqrt(outputLocal, inputLocal, dataSize);

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(int16_t));
    LocalTensor<int16_t> inputLocal2 = tbuf2.Get<int16_t>();

    TBuf<TPosition::VECCALC> tbuf3;
    tpipe.InitBuffer(tbuf3, dataSize * 2 * sizeof(int8_t));
    LocalTensor<int8_t> outputLocal2 = tbuf3.Get<int8_t>();

    TBuf<TPosition::VECCALC> tbuf4;
    tpipe.InitBuffer(tbuf4, dataSize * 2 * sizeof(uint8_t));
    LocalTensor<uint8_t> outputLocal3 = tbuf4.Get<uint8_t>();
    if constexpr (USE_CAST_DEQ){
        CastDeq<int8_t, int16_t, false, false>(outputLocal2, inputLocal2, dataSize);
        CastDeq<uint8_t, int16_t, false, true>(outputLocal3, inputLocal2, dataSize);
        CastDeq<int8_t, int16_t, true, false>(outputLocal2, inputLocal2, dataSize);
        CastDeq<uint8_t, int16_t, true, true>(outputLocal3, inputLocal2, dataSize);
        int maskCount = 32;
        UnaryRepeatParams repeatParams;
        CastDeq<int8_t, int16_t, true, false, false>(outputLocal2, inputLocal2, maskCount, 2, repeatParams);
        CastDeq<uint8_t, int16_t, true, false, true>(outputLocal3, inputLocal2, maskCount, 2, repeatParams);
        CastDeq<int8_t, int16_t, true, true, false>(outputLocal2, inputLocal2, maskCount, 2, repeatParams);
        CastDeq<uint8_t, int16_t, true, true, true>(outputLocal3, inputLocal2, maskCount, 2, repeatParams);
        uint64_t maskVec[2] = {0xff, 0};
        CastDeq<int8_t, int16_t, true, false, false>(outputLocal2, inputLocal2, maskVec, 2, repeatParams);
        CastDeq<uint8_t, int16_t, true, false, true>(outputLocal3, inputLocal2, maskVec, 2, repeatParams);
        CastDeq<int8_t, int16_t, true, true, false>(outputLocal2, inputLocal2, maskVec, 2, repeatParams);
        CastDeq<uint8_t, int16_t, true, true, true>(outputLocal3, inputLocal2, maskVec, 2, repeatParams);
    }else{
        CastDequant<int8_t, int16_t, false, false>(outputLocal2, inputLocal2, dataSize);
        CastDequant<uint8_t, int16_t, false, true>(outputLocal3, inputLocal2, dataSize);
        CastDequant<int8_t, int16_t, true, false>(outputLocal2, inputLocal2, dataSize);
        CastDequant<uint8_t, int16_t, true, true>(outputLocal3, inputLocal2, dataSize);
        int maskCount = 32;
        UnaryRepeatParams repeatParams;
        CastDequant<int8_t, int16_t, true, false, false>(outputLocal2, inputLocal2, maskCount, 2, repeatParams);
        CastDequant<uint8_t, int16_t, true, false, true>(outputLocal3, inputLocal2, maskCount, 2, repeatParams);
        CastDequant<int8_t, int16_t, true, true, false>(outputLocal2, inputLocal2, maskCount, 2, repeatParams);
        CastDequant<uint8_t, int16_t, true, true, true>(outputLocal3, inputLocal2, maskCount, 2, repeatParams);
        uint64_t maskVec[2] = {0xff, 0};
        CastDequant<int8_t, int16_t, true, false, false>(outputLocal2, inputLocal2, maskVec, 2, repeatParams);
        CastDequant<uint8_t, int16_t, true, false, true>(outputLocal3, inputLocal2, maskVec, 2, repeatParams);
        CastDequant<int8_t, int16_t, true, true, false>(outputLocal2, inputLocal2, maskVec, 2, repeatParams);
        CastDequant<uint8_t, int16_t, true, true, true>(outputLocal3, inputLocal2, maskVec, 2, repeatParams);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
void MainVecUnary02(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
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
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Relu(outputLocal, inputLocal, dataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
void MainVecUnary03(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
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
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Not(outputLocal, inputLocal, dataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
void MainVecUnary04(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
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
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Abs(outputLocal, inputLocal, dataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

struct UnaryTestParams {
    int32_t dataSize;
    int32_t dataBitSize;
    void (*cal_func)(uint8_t*, uint8_t*, int32_t);
};

class UnarySimpleTestsuite : public testing::Test, public testing::WithParamInterface<UnaryTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_UNARY_SIMPLE, UnarySimpleTestsuite,
    ::testing::Values(UnaryTestParams { 208, 2, MainVecUnary01<half, true> },
    UnaryTestParams { 208, 4, MainVecUnary01<float, true> }, UnaryTestParams { 208, 2, MainVecUnary01<half, false> },
    UnaryTestParams { 208, 4, MainVecUnary01<float, false> }, UnaryTestParams { 208, 4, MainVecUnary02<int32_t> },
    UnaryTestParams { 208, 4, MainVecUnary02<half> }, UnaryTestParams { 208, 4, MainVecUnary02<float> },
    UnaryTestParams { 208, 2, MainVecUnary03<int16_t> }, UnaryTestParams { 208, 2, MainVecUnary03<uint16_t> },
    UnaryTestParams { 208, 4, MainVecUnary04<float> }));

TEST_P(UnarySimpleTestsuite, UnarySimpleTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.dataBitSize];
    uint8_t dstGm[param.dataSize * param.dataBitSize];

    param.cal_func(srcGm, dstGm, param.dataSize);
    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

class TEST_VEC_UNARY_F : public testing::Test {
protected:
    void SetUp() {}

    void TearDown() {}
};

TEST_F(TEST_VEC_UNARY_F, VecUnaryCase)
{
    int32_t dataSize = 128;
    int16_t src[dataSize * sizeof(int16_t)];
    int16_t dst[dataSize * sizeof(int16_t)];
    uint16_t AbsSrcBlkStride = 1;
    uint16_t AbsDstBlkStride = 1;
    uint16_t AbsSrcRepStride = 8;
    uint16_t AbsDstRepStride = 8;
    set_vector_mask(UINT64_MAX, UINT64_MAX);
    vabs(dst, src, 1, AbsDstBlkStride, AbsSrcBlkStride, AbsDstRepStride, AbsSrcRepStride);
    for (int32_t i = 0; i < dataSize; i++) {
        EXPECT_EQ(dst[i], 0x0000);
    }
}
