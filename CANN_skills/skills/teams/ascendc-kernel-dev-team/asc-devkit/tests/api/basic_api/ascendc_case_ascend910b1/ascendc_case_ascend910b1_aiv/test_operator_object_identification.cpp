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
#include "kernel_utils.h"
// #include "model/model_factory.h"
#include "common.h"
#include <math.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace AscendC;
template <typename T>
__global__ __aicore__ void MainMrgSort(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm)
{
    TPipe tpipe;
    int32_t dataSize = 512;
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

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    MrgSort4Info params;
    params.elementLengths[0] = 32;
    params.elementLengths[1] = 32;
    params.elementLengths[2] = 32;
    params.elementLengths[3] = 32;
    params.ifExhaustedSuspension = false;
    params.validBit = 0b1111;
    params.repeatTimes = 1;

    MrgSortSrcList<T> srcLocal;
    srcLocal.src1 = inputLocal[0];
    srcLocal.src2 = inputLocal[64];
    srcLocal.src3 = inputLocal[128];
    srcLocal.src4 = inputLocal[192];

    MrgSort<T>(outputLocal, srcLocal, params);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

struct MrgSort4TestParams {
    int32_t typeSize;
    void (*CalFunc)(uint8_t*, uint8_t*);
};

class MrgSort4Testsuite : public testing::Test, public testing::WithParamInterface<MrgSort4TestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_MRGSORT4, MrgSort4Testsuite,
    ::testing::Values(MrgSort4TestParams { 2, MainMrgSort<half> }, MrgSort4TestParams { 4, MainMrgSort<float> }));

TEST_P(MrgSort4Testsuite, MrgSortOpTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[512 * param.typeSize];
    uint8_t dstGm[512 * 4];

    param.CalFunc(dstGm, srcGm);
}

template <typename T>
__global__ __aicore__ void MainSort32(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ src1Gm)
{
    TPipe tpipe;
    int32_t dataSize = 128;
    int32_t repeat = dataSize / 32;

    GlobalTensor<T> inputGlobal0;
    GlobalTensor<uint32_t> inputGlobal1;
    GlobalTensor<T> outputGlobal;
    inputGlobal0.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    inputGlobal1.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(src1Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize * 2);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> inputLocal0 = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(uint32_t));
    LocalTensor<uint32_t> inputLocal1 = tbuf1.Get<uint32_t>();
    
    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, (dataSize * 2 + 2 * dataSize * (sizeof(uint32_t) - sizeof(T))) * sizeof(T));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    DataCopy(inputLocal0, inputGlobal0, dataSize);

    DataCopy<uint32_t>(inputLocal1, inputGlobal1, dataSize);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Sort32<T>(outputLocal, inputLocal0, inputLocal1, repeat);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy<T>(outputGlobal, outputLocal, dataSize * 2);

    pipe_barrier(PIPE_ALL);
}

struct Sort32TestParams {
    int32_t typeSize;
    void (*CalFunc)(uint8_t*, uint8_t*, uint8_t*);
};

class Sort32Testsuite : public testing::Test, public testing::WithParamInterface<Sort32TestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_SORT32, Sort32Testsuite,
    ::testing::Values(Sort32TestParams { 2, MainSort32<half> }, Sort32TestParams { 4, MainSort32<float> }));

TEST_P(Sort32Testsuite, Sort32OpTestCase)
{
    auto param = GetParam();
    uint8_t src0Gm[128 * param.typeSize];
    uint8_t src1Gm[128 * 4];
    uint8_t dstGm[256 * 4];

    param.CalFunc(dstGm, src0Gm, src1Gm);
}
