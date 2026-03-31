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
struct SelfTensorDesc : public TensorTrait<T> {
    static constexpr int32_t tensorLen = 256;
};

template <typename T>
void MainVecBinaryOperatorSimpleDemo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ src1Gm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<SelfTensorDesc<T>> input0Global;
    GlobalTensor<SelfTensorDesc<T>> input1Global;
    GlobalTensor<SelfTensorDesc<T>> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input0Local = tbuf.Get<SelfTensorDesc<T>>();
    input0Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input1Local = tbuf1.Get<SelfTensorDesc<T>>();
    input1Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> outputLocal = tbuf2.Get<SelfTensorDesc<T>>();
    outputLocal.SetSize(dataSize);

    for (int32_t i = 0; i < dataSize; ++i) {
        input1Local.SetValue(i, 1);
    }

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    BinaryRepeatParams binaryParams;
    Add(outputLocal, input0Local, input1Local, dataSize);
    outputLocal = input0Local + input1Local;
    Sub(outputLocal, outputLocal, input1Local, dataSize);
    outputLocal = outputLocal - input1Local;
    Mul(outputLocal, outputLocal, input1Local, dataSize);
    outputLocal = outputLocal * input1Local;
    Max(outputLocal, outputLocal, input1Local, dataSize);
    Min(outputLocal, outputLocal, input1Local, dataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(outputGlobal, outputLocal, dataSize);
    SetAtomicNone();
    pipe_barrier(PIPE_ALL);
}

template <typename T>
void MainVecBinaryOperatorSimpleErrorDemo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ src1Gm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<SelfTensorDesc<T>> input0Global;
    GlobalTensor<SelfTensorDesc<T>> input1Global;
    GlobalTensor<SelfTensorDesc<T>> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input0Local = tbuf.Get<SelfTensorDesc<T>>();
    input0Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input1Local = tbuf1.Get<SelfTensorDesc<T>>();
    input1Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> outputLocal = tbuf2.Get<SelfTensorDesc<T>>();
    outputLocal.SetSize(dataSize);

    for (int32_t i = 0; i < dataSize; ++i) {
        input1Local.SetValue(i, 1);
    }

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    BinaryRepeatParams binaryParams;
    uint64_t mask = 256 / sizeof(T);
    uint64_t maskbit[2] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
    int32_t dataSizeTmp = 512;
    Add(outputLocal, input0Local, input1Local, dataSizeTmp);
    Add(outputLocal, input0Local, input1Local, mask, dataSizeTmp / mask, binaryParams);
    Add(outputLocal, input0Local, input1Local, maskbit, dataSizeTmp / mask, binaryParams);
    outputLocal = input0Local + input1Local;
    Sub(outputLocal, outputLocal, input1Local, dataSizeTmp);
    Sub(outputLocal, input0Local, input1Local, mask, dataSizeTmp / mask, binaryParams);
    Sub(outputLocal, input0Local, input1Local, maskbit, dataSizeTmp / mask, binaryParams);
    outputLocal = outputLocal - input1Local;
    Mul(outputLocal, outputLocal, input1Local, dataSizeTmp);
    Mul(outputLocal, input0Local, input1Local, mask, dataSizeTmp / mask, binaryParams);
    Mul(outputLocal, input0Local, input1Local, maskbit, dataSizeTmp / mask, binaryParams);
    outputLocal = outputLocal * input1Local;
    Max(outputLocal, outputLocal, input1Local, dataSizeTmp);
    Max(outputLocal, input0Local, input1Local, mask, dataSizeTmp / mask, binaryParams);
    Max(outputLocal, input0Local, input1Local, maskbit, dataSizeTmp / mask, binaryParams);
    Min(outputLocal, outputLocal, input1Local, dataSizeTmp);
    Min(outputLocal, input0Local, input1Local, mask, dataSizeTmp / mask, binaryParams);
    Min(outputLocal, input0Local, input1Local, maskbit, dataSizeTmp / mask, binaryParams);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(outputGlobal, outputLocal, dataSize);
    SetAtomicNone();
    pipe_barrier(PIPE_ALL);
}


template <typename T>
void MainVecBinaryOperatorDivDemo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ src1Gm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<SelfTensorDesc<T>> input0Global;
    GlobalTensor<SelfTensorDesc<T>> input1Global;
    GlobalTensor<SelfTensorDesc<T>> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input0Local = tbuf.Get<SelfTensorDesc<T>>();
    input0Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input1Local = tbuf1.Get<SelfTensorDesc<T>>();
    input1Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> outputLocal = tbuf2.Get<SelfTensorDesc<T>>();
    outputLocal.SetSize(dataSize);

    for (int32_t i = 0; i < dataSize; ++i) {
        input1Local.SetValue(i, 1);
    }

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Div(outputLocal, input0Local, input1Local, dataSize);
    outputLocal = input0Local / input1Local;
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
void MainVecBinaryOperatorDivErrorDemo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ src1Gm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<SelfTensorDesc<T>> input0Global;
    GlobalTensor<SelfTensorDesc<T>> input1Global;
    GlobalTensor<SelfTensorDesc<T>> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input0Local = tbuf.Get<SelfTensorDesc<T>>();
    input0Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input1Local = tbuf1.Get<SelfTensorDesc<T>>();
    input1Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> outputLocal = tbuf2.Get<SelfTensorDesc<T>>();
    outputLocal.SetSize(dataSize);

    for (int32_t i = 0; i < dataSize; ++i) {
        input1Local.SetValue(i, 1);
    }

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    BinaryRepeatParams binaryParams;
    uint64_t mask = 256 / sizeof(T);
    uint64_t maskbit[2] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
    int32_t dataSizeTmp = 512;
    Div(outputLocal, input0Local, input1Local, dataSizeTmp);
    Div(outputLocal, input0Local, input1Local, mask, dataSizeTmp / mask, binaryParams);
    Div(outputLocal, input0Local, input1Local, maskbit, dataSizeTmp / mask, binaryParams);
    outputLocal = input0Local / input1Local;
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
void MainVecBinaryOperatorAndOrDemo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ src1Gm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<SelfTensorDesc<T>> input0Global;
    GlobalTensor<SelfTensorDesc<T>> input1Global;
    GlobalTensor<SelfTensorDesc<T>> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input0Local = tbuf.Get<SelfTensorDesc<T>>();
    input0Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input1Local = tbuf1.Get<SelfTensorDesc<T>>();
    input1Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> outputLocal = tbuf2.Get<SelfTensorDesc<T>>();
    outputLocal.SetSize(dataSize);

    for (int32_t i = 0; i < dataSize; ++i) {
        input1Local.SetValue(i, 1);
    }

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    And(outputLocal, input0Local, input1Local, dataSize);
    outputLocal = input0Local & input1Local;
    Or(outputLocal, outputLocal, input1Local, dataSize);
    outputLocal = outputLocal | input1Local;
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
void MainVecBinaryOperatorAndOrErrorDemo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ src1Gm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<SelfTensorDesc<T>> input0Global;
    GlobalTensor<SelfTensorDesc<T>> input1Global;
    GlobalTensor<SelfTensorDesc<T>> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input0Local = tbuf.Get<SelfTensorDesc<T>>();
    input0Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input1Local = tbuf1.Get<SelfTensorDesc<T>>();
    input1Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> outputLocal = tbuf2.Get<SelfTensorDesc<T>>();
    outputLocal.SetSize(dataSize);

    for (int32_t i = 0; i < dataSize; ++i) {
        input1Local.SetValue(i, 1);
    }

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    BinaryRepeatParams binaryParams;
    uint64_t mask = 256 / sizeof(T);
    uint64_t maskbit[2] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
    int32_t dataSizeTmp = 512;

    And(outputLocal, input0Local, input1Local, dataSizeTmp);
    And(outputLocal, input0Local, input1Local, mask, dataSizeTmp / mask, binaryParams);
    And(outputLocal, input0Local, input1Local, maskbit, dataSizeTmp / mask, binaryParams);
    outputLocal = input0Local & input1Local;
    Or(outputLocal, outputLocal, input1Local, dataSizeTmp);
    Or(outputLocal, input0Local, input1Local, mask, dataSizeTmp / mask, binaryParams);
    Or(outputLocal, input0Local, input1Local, maskbit, dataSizeTmp / mask, binaryParams);
    outputLocal = outputLocal | input1Local;
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
void MainVecBinaryOperatorAddreluDemo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ src1Gm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<SelfTensorDesc<T>> input0Global;
    GlobalTensor<SelfTensorDesc<T>> input1Global;
    GlobalTensor<SelfTensorDesc<T>> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input0Local = tbuf.Get<SelfTensorDesc<T>>();
    input0Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input1Local = tbuf1.Get<SelfTensorDesc<T>>();
    input1Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> outputLocal = tbuf2.Get<SelfTensorDesc<T>>();
    outputLocal.SetSize(dataSize);

    for (int32_t i = 0; i < dataSize; ++i) {
        input1Local.SetValue(i, 1);
    }

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AddRelu(outputLocal, outputLocal, input1Local, dataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(outputGlobal, outputLocal, dataSize);
    SetAtomicNone();
    pipe_barrier(PIPE_ALL);
}

template <typename T>
void MainVecBinaryOperatorAddreluSubreluErrorDemo(__gm__ uint8_t* __restrict__ dstGm,
    __gm__ uint8_t* __restrict__ src0Gm, __gm__ uint8_t* __restrict__ src1Gm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<SelfTensorDesc<T>> input0Global;
    GlobalTensor<SelfTensorDesc<T>> input1Global;
    GlobalTensor<SelfTensorDesc<T>> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input0Local = tbuf.Get<SelfTensorDesc<T>>();
    input0Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input1Local = tbuf1.Get<SelfTensorDesc<T>>();
    input1Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> outputLocal = tbuf2.Get<SelfTensorDesc<T>>();
    outputLocal.SetSize(dataSize);

    for (int32_t i = 0; i < dataSize; ++i) {
        input1Local.SetValue(i, 1);
    }

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    BinaryRepeatParams binaryParams;
    uint64_t mask = 256 / sizeof(T);
    uint64_t maskbit[2] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
    int32_t dataSizeTmp = 512;
    AddRelu(outputLocal, outputLocal, input1Local, dataSizeTmp);
    AddRelu(outputLocal, input0Local, input1Local, mask, dataSizeTmp / mask, binaryParams);
    AddRelu(outputLocal, input0Local, input1Local, maskbit, dataSizeTmp / mask, binaryParams);
    SubRelu(outputLocal, outputLocal, input1Local, dataSizeTmp);
    SubRelu(outputLocal, input0Local, input1Local, mask, dataSizeTmp / mask, binaryParams);
    SubRelu(outputLocal, input0Local, input1Local, maskbit, dataSizeTmp / mask, binaryParams);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(outputGlobal, outputLocal, dataSize);
    SetAtomicNone();
    pipe_barrier(PIPE_ALL);
}

template <typename T>
void MainVecBinaryOperatorSubreluDemo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ src1Gm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<SelfTensorDesc<T>> input0Global;
    GlobalTensor<SelfTensorDesc<T>> input1Global;
    GlobalTensor<SelfTensorDesc<T>> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input0Local = tbuf.Get<SelfTensorDesc<T>>();
    input0Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> input1Local = tbuf1.Get<SelfTensorDesc<T>>();
    input1Local.SetSize(dataSize);

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<SelfTensorDesc<T>> outputLocal = tbuf2.Get<SelfTensorDesc<T>>();
    outputLocal.SetSize(dataSize);

    for (int32_t i = 0; i < dataSize; ++i) {
        input1Local.SetValue(i, 1);
    }

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    SubRelu(outputLocal, outputLocal, input1Local, dataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(outputGlobal, outputLocal, dataSize);
    SetAtomicNone();
    pipe_barrier(PIPE_ALL);
}

struct BinaryDescTestParams {
    int32_t dataSize;
    int32_t dataBitSize;
    bool expectRes;
    int32_t errorTimes;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, int32_t);
};

class BinaryDescSimpleTestsuite : public testing::Test, public testing::WithParamInterface<BinaryDescTestParams> {
protected:
    void SetUp() {}
    void TearDown() {GlobalMockObject::verify();}
};

INSTANTIATE_TEST_CASE_P(TEST_VEC_BINARY_DESC_SIMPLE, BinaryDescSimpleTestsuite,
    ::testing::Values(BinaryDescTestParams { 256, 2, true, 0, MainVecBinaryOperatorSimpleDemo<half> },
    BinaryDescTestParams { 256, 4, true, 0, MainVecBinaryOperatorSimpleDemo<float> },
    BinaryDescTestParams { 256, 2, true, 0, MainVecBinaryOperatorSimpleDemo<int16_t> },
    BinaryDescTestParams { 256, 4, true, 0, MainVecBinaryOperatorSimpleDemo<int32_t> },

    BinaryDescTestParams { 256, 4, true, 0, MainVecBinaryOperatorDivDemo<half> },
    BinaryDescTestParams { 256, 4, true, 0, MainVecBinaryOperatorDivDemo<float> },

    BinaryDescTestParams { 256, 4, true, 0, MainVecBinaryOperatorAndOrDemo<uint16_t> },
    BinaryDescTestParams { 256, 4, true, 0, MainVecBinaryOperatorAndOrDemo<int16_t> },
    BinaryDescTestParams { 256, 4, true, 0, MainVecBinaryOperatorAddreluDemo<half> },
    BinaryDescTestParams { 256, 4, true, 0, MainVecBinaryOperatorAddreluDemo<float> },
    BinaryDescTestParams { 256, 4, true, 0, MainVecBinaryOperatorAddreluDemo<int16_t> },
    BinaryDescTestParams { 256, 4, true, 0, MainVecBinaryOperatorSubreluDemo<half> },
    BinaryDescTestParams { 256, 4, true, 0, MainVecBinaryOperatorSubreluDemo<float> },
    BinaryDescTestParams { 256, 4, true, 0, MainVecBinaryOperatorSubreluDemo<int16_t>},

    BinaryDescTestParams { 256, 4, false, 15, MainVecBinaryOperatorSimpleErrorDemo<int32_t> }, // Error check
    BinaryDescTestParams { 256, 4, false, 3, MainVecBinaryOperatorDivErrorDemo<float> },
    BinaryDescTestParams { 256, 4, false, 6, MainVecBinaryOperatorAndOrErrorDemo<int16_t> },
    BinaryDescTestParams { 256, 4, false, 6, MainVecBinaryOperatorAddreluSubreluErrorDemo<float>}
));

TEST_P(BinaryDescSimpleTestsuite, BinaryDescSimpleTestCase)
{

    TPipe tpipe;
    AscendC::SetGCoreType(2);
    auto param = GetParam();
    uint8_t src0Gm[param.dataSize * param.dataBitSize];
    uint8_t src1Gm[param.dataSize * param.dataBitSize];
    uint8_t dstGm[param.dataSize * param.dataBitSize];
    if (!param.expectRes) {
        MOCKER(raise, int(*)(int)).times(param.errorTimes).will(returnValue(0));
    }
    param.cal_func(dstGm, src0Gm, src1Gm, param.dataSize);
    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
    AscendC::SetGCoreType(0);
}