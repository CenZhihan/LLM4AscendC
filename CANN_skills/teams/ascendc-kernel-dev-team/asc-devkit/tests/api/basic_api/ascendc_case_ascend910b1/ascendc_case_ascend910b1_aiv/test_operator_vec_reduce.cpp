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
void MainVecReduce(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm,
    __gm__ int32_t srcDataSize, __gm__ int32_t dstDataSize)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(srcGm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(dstGm), dstDataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, srcDataSize * sizeof(PrimT<T>));
    LocalTensor<T> inputLocal = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dstDataSize * sizeof(PrimT<T>));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    AscendCUtils::SetMask<uint8_t>(256);
    DataCopy(inputLocal, inputGlobal, srcDataSize);

    AscendCUtils::SetMask<uint8_t>(256);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    BlockReduceMax<T>(outputLocal, inputLocal, 1, 64, 8, 1, 8);
    BlockReduceMin<T>(outputLocal, inputLocal, 1, 64, 8, 1, 8);
    BlockReduceSum<T>(outputLocal, inputLocal, 1, 64, 8, 1, 8);
    PairReduceSum<T>(outputLocal, inputLocal, 1, 64, 8, 1, 8);

    SetMaskCount();
    SetVectorMask<PrimT<T>, MaskMode::COUNTER>(0, 64);
    BlockReduceMax<T, false>(outputLocal, inputLocal, 1, 64, 8, 1, 8);
    BlockReduceMin<T, false>(outputLocal, inputLocal, 1, 64, 8, 1, 8);
    BlockReduceSum<T, false>(outputLocal, inputLocal, 1, 64, 8, 1, 8);
    PairReduceSum<T, false>(outputLocal, inputLocal, 1, 64, 8, 1, 8);
    SetMaskNorm();
    AscendCUtils::ResetMask();
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    PipeBarrier<PIPE_ALL>();
}

class TEST_VEC_REDUCE : public testing::Test {
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

TEST_F(TEST_VEC_REDUCE, OperatorVecReduceFloatCase)
{
    const int32_t srcDataSize = 128;
    const int32_t dstDataSize = 64;

    uint8_t srcGm[srcDataSize * sizeof(float)];
    uint8_t dstGm[dstDataSize * sizeof(float)];

    MainVecReduce<float>(srcGm, dstGm, srcDataSize, dstDataSize);

    for (int32_t i = 0; i < dstDataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

TEST_F(TEST_VEC_REDUCE, OperatorVecReduceHalfCase)
{
    const int32_t srcDataSize = 128;
    const int32_t dstDataSize = 64;

    uint8_t srcGm[srcDataSize * sizeof(half)];
    uint8_t dstGm[dstDataSize * sizeof(half)];

    MainVecReduce<half>(srcGm, dstGm, srcDataSize, dstDataSize);

    for (int32_t i = 0; i < dstDataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

TEST_F(TEST_VEC_REDUCE, OperatorVecReduceTensorTraitFloatCase)
{
    const int32_t srcDataSize = 128;
    const int32_t dstDataSize = 64;

    uint8_t srcGm[srcDataSize * sizeof(float)];
    uint8_t dstGm[dstDataSize * sizeof(float)];

    MainVecReduce<TensorTrait<float>>(srcGm, dstGm, srcDataSize, dstDataSize);

    for (int32_t i = 0; i < dstDataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

TEST_F(TEST_VEC_REDUCE, OperatorVecReduceTensorTraitHalfCase)
{
    const int32_t srcDataSize = 128;
    const int32_t dstDataSize = 64;

    uint8_t srcGm[srcDataSize * sizeof(half)];
    uint8_t dstGm[dstDataSize * sizeof(half)];

    MainVecReduce<TensorTrait<half>>(srcGm, dstGm, srcDataSize, dstDataSize);

    for (int32_t i = 0; i < dstDataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

template <typename T>
__global__ __aicore__ void MainAllReduceSimple(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm,
    __gm__ int32_t srcDataSize, __gm__ int32_t dstDataSize, __gm__ int32_t level, int32_t count)
{
    TPipe tpipe;
    int32_t mask = 64;
    uint64_t masks[2]{ FULL_MASK, 0 };
    if (sizeof(PrimT<T>) == sizeof(half)) {
        mask = 128;
        masks[0] = FULL_MASK;
        masks[1] = FULL_MASK;
    }
    int32_t repStride = 8;
    int32_t repeat = srcDataSize / mask;

    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(srcGm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(dstGm), dstDataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, srcDataSize * sizeof(PrimT<T>));
    LocalTensor<T> inputLocal = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dstDataSize * sizeof(PrimT<T>));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    TBuf<TPosition::VECCALC> tbuf3;
    tpipe.InitBuffer(tbuf3, repeat * repStride);
    LocalTensor<T> workLocal = tbuf3.Get<T>();

    DataCopy(inputLocal, inputGlobal, srcDataSize);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    switch (level) {
        case 0:
            ReduceMax<T>(outputLocal, inputLocal, workLocal, mask, repeat, repStride, false);
            ReduceMax<T>(outputLocal, inputLocal, workLocal, mask, repeat, repStride, true);
            ReduceMin<T>(outputLocal, inputLocal, workLocal, mask, repeat, repStride, false);
            ReduceMin<T>(outputLocal, inputLocal, workLocal, mask, repeat, repStride, true);
            ReduceSum<T>(outputLocal, inputLocal, workLocal, mask, repeat, repStride);

            ReduceMax<T>(outputLocal, inputLocal, workLocal, masks, repeat, repStride, false);
            ReduceMax<T>(outputLocal, inputLocal, workLocal, masks, repeat, repStride, true);
            ReduceMin<T>(outputLocal, inputLocal, workLocal, masks, repeat, repStride, false);
            ReduceMin<T>(outputLocal, inputLocal, workLocal, masks, repeat, repStride, true);
            ReduceSum<T>(outputLocal, inputLocal, workLocal, masks, repeat, repStride);
            break;
        case 2:
            ReduceMax<T>(outputLocal, inputLocal, workLocal, count, false);
            ReduceMax<T>(outputLocal, inputLocal, workLocal, count, true);
            ReduceMin<T>(outputLocal, inputLocal, workLocal, count, false);
            ReduceMin<T>(outputLocal, inputLocal, workLocal, count, true);
            ReduceSum<T>(outputLocal, inputLocal, workLocal, count);
            break;
        default:
            break;
    }
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__global__ __aicore__ void MainWholeReduceSimple(__gm__ uint8_t* __restrict__ srcGm,
    __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t srcDataSize, __gm__ int32_t dstDataSize,
    __gm__ int32_t level, int32_t count)
{
    TPipe tpipe;
    int32_t mask = 64;
    uint64_t masks[2]{ FULL_MASK, 0 };
    if (sizeof(PrimT<T>) == sizeof(half)) {
        mask = 128;
        masks[0] = FULL_MASK;
        masks[1] = FULL_MASK;
    }
    int32_t repStride = 8;
    int32_t repeat = srcDataSize / mask;

    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(srcGm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(dstGm), dstDataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, srcDataSize * sizeof(PrimT<T>));
    LocalTensor<T> inputLocal = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dstDataSize * sizeof(PrimT<T>));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    TBuf<TPosition::VECCALC> tbuf3;
    tpipe.InitBuffer(tbuf3, repeat * repStride);
    LocalTensor<T> workLocal = tbuf3.Get<T>();

    DataCopy(inputLocal, inputGlobal, srcDataSize);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    WholeReduceMax<T>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);
    WholeReduceMax<T>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_INDEX_VALUE);
    WholeReduceMax<T>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_VALUE);
    WholeReduceMax<T>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_INDEX);
    WholeReduceSum<T>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride);
    WholeReduceMin<T>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);
    WholeReduceMin<T>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_INDEX_VALUE);
    WholeReduceMin<T>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_VALUE);
    WholeReduceMin<T>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_INDEX);

    WholeReduceMax<T>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);
    WholeReduceMax<T>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_INDEX_VALUE);
    WholeReduceMax<T>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_VALUE);
    WholeReduceMax<T>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_INDEX);
    WholeReduceSum<T>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride);
    WholeReduceMin<T>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);
    WholeReduceMin<T>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_INDEX_VALUE);
    WholeReduceMin<T>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_VALUE);
    WholeReduceMin<T>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_INDEX);

    SetMaskCount();
    SetVectorMask<PrimT<T>, MaskMode::COUNTER>(0, 64);
    WholeReduceMax<T, false>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);
    WholeReduceMax<T, false>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_INDEX_VALUE);
    WholeReduceMax<T, false>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_VALUE);
    WholeReduceMax<T, false>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_INDEX);
    WholeReduceSum<T, false>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride);
    WholeReduceMin<T, false>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);
    WholeReduceMin<T, false>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_INDEX_VALUE);
    WholeReduceMin<T, false>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_VALUE);
    WholeReduceMin<T, false>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_INDEX);

    WholeReduceMax<T, false>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);
    WholeReduceMax<T, false>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_INDEX_VALUE);
    WholeReduceMax<T, false>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_VALUE);
    WholeReduceMax<T, false>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_INDEX);
    WholeReduceSum<T, false>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride);
    WholeReduceMin<T, false>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);
    WholeReduceMin<T, false>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_INDEX_VALUE);
    WholeReduceMin<T, false>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_VALUE);
    WholeReduceMin<T, false>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_ONLY_INDEX);
    SetMaskNorm();
    AscendCUtils::ResetMask();
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    PipeBarrier<PIPE_ALL>();
}
struct ReduceTestParams {
    void (*calFunc)(uint8_t*, uint8_t*, int32_t, int32_t, int32_t, int32_t);
    int32_t srcDataSize;
    int32_t dstDataSize;
    int32_t dtypeSize;
    int32_t level;
    int32_t count;
};

class ReduceSimpleTestsuite : public testing::Test, public testing::WithParamInterface<ReduceTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_REDUCE_SIMPLE, ReduceSimpleTestsuite,
    ::testing::Values(ReduceTestParams{ MainAllReduceSimple<half>, 128, 16, 2, 0, 128 },
    ReduceTestParams{ MainAllReduceSimple<float>, 64, 16, 4, 0, 64 },
    ReduceTestParams{ MainAllReduceSimple<float>, 64, 16, 4, 2, 64 },
    ReduceTestParams{ MainAllReduceSimple<float>, 64, 16, 4, 2, 63 },
    ReduceTestParams{ MainAllReduceSimple<float>, 16648, 16, 4, 0, 16648 },
    ReduceTestParams{ MainAllReduceSimple<float>, 16648, 16, 4, 2, 16648 },
    ReduceTestParams{ MainWholeReduceSimple<float>, 64, 16, 4, 0, 64},
    ReduceTestParams{ MainAllReduceSimple<half>, 128, 16, 2, 0, 128 },
    ReduceTestParams{ MainAllReduceSimple<half>, 128, 16, 2, 2, 128 },
    ReduceTestParams{ MainAllReduceSimple<half>, 32656, 16, 2, 0, 32656 },
    ReduceTestParams{ MainAllReduceSimple<half>, 32656, 16, 2, 2, 32656 },
    ReduceTestParams{ MainWholeReduceSimple<half>, 128, 16, 2, 0, 128 },
    // TensorTrait Case
    ReduceTestParams{ MainAllReduceSimple<TensorTrait<float>>, 16648, 16, 4, 2, 16648 },
    ReduceTestParams{ MainAllReduceSimple<TensorTrait<half>>, 32656, 16, 2, 2, 32656 },
    ReduceTestParams{ MainWholeReduceSimple<TensorTrait<float>>, 64, 16, 4, 0, 64 },
    ReduceTestParams{ MainWholeReduceSimple<TensorTrait<half>>, 128, 16, 2, 0, 128 }
    ));

TEST_P(ReduceSimpleTestsuite, ReduceSimpleTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.srcDataSize * param.dtypeSize];
    uint8_t dstGm[param.dstDataSize * param.dtypeSize];

    param.calFunc(srcGm, dstGm, param.srcDataSize, param.dstDataSize, param.level, param.count);
    for (int32_t i = 0; i < param.dstDataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
TEST_F(ReduceSimpleTestsuite, GetMaxMinCntCase)
{
    int32_t srcDataSize = 64;
    int32_t dstDataSize = 16;
    int32_t dtypeSize = 4;

    uint8_t srcGm[srcDataSize * dtypeSize];
    uint8_t dstGm[dstDataSize * dtypeSize];
    TPipe tpipe;
    int32_t mask = 64;
    int32_t repStride = 8;
    int32_t repeat = srcDataSize / mask;

    GlobalTensor<float> inputGlobal;
    GlobalTensor<float> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(srcGm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstGm), dstDataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, srcDataSize * sizeof(float));
    LocalTensor<float> inputLocal = tbuf1.Get<float>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dstDataSize * sizeof(float));
    LocalTensor<float> outputLocal = tbuf2.Get<float>();

    TBuf<TPosition::VECCALC> tbuf3;
    tpipe.InitBuffer(tbuf3, repeat * repStride);
    LocalTensor<float> workLocal = tbuf3.Get<float>();

    DataCopy(inputLocal, inputGlobal, srcDataSize);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    WholeReduceMax<float>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    uint32_t maxMinVal = 0;
    uint32_t maxMinIdx = 0;
    GetReduceMaxMinCount<float>(maxMinVal, maxMinIdx);
    EXPECT_EQ(maxMinVal, 0x0);
    EXPECT_EQ(maxMinIdx, 0x0);

    float maxMinFp32Val = 0;
    float maxMinFp32Idx = 0;
    GetReduceMaxMinCount<float>(maxMinFp32Val, maxMinFp32Idx);
    EXPECT_EQ(maxMinFp32Val, 0x0);
    EXPECT_EQ(maxMinFp32Idx, 0x0);

    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    PipeBarrier<PIPE_ALL>();
}
TEST_F(ReduceSimpleTestsuite, GetReduceRepeatMaxMinSprCase)
{
    int32_t srcDataSize = 64;
    int32_t dstDataSize = 16;
    int32_t dtypeSize = 4;

    uint8_t srcGm[srcDataSize * dtypeSize];
    uint8_t dstGm[dstDataSize * dtypeSize];
    TPipe tpipe;
    int32_t mask = 64;
    int32_t repStride = 8;
    int32_t repeat = srcDataSize / mask;

    GlobalTensor<float> inputGlobal;
    GlobalTensor<float> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(srcGm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstGm), dstDataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, srcDataSize * sizeof(float));
    LocalTensor<float> inputLocal = tbuf1.Get<float>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dstDataSize * sizeof(float));
    LocalTensor<float> outputLocal = tbuf2.Get<float>();

    TBuf<TPosition::VECCALC> tbuf3;
    tpipe.InitBuffer(tbuf3, repeat * repStride);
    LocalTensor<float> workLocal = tbuf3.Get<float>();

    DataCopy(inputLocal, inputGlobal, srcDataSize);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    WholeReduceMax<float>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    uint32_t maxMinVal = 0;
    uint32_t maxMinIdx = 0;
    GetReduceRepeatMaxMinSpr<float>(maxMinVal, maxMinIdx);
    EXPECT_EQ(maxMinVal, 0x0);
    EXPECT_EQ(maxMinIdx, 0x0);

    float maxMinFp32Val = 0;
    float maxMinFp32Idx = 0;
    GetReduceRepeatMaxMinSpr<float>(maxMinFp32Val, maxMinFp32Idx);
    EXPECT_EQ(maxMinFp32Val, 0x0);
    EXPECT_EQ(maxMinFp32Idx, 0x0);

    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__global__ __aicore__ void MainRepeatReduceSum(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm,
    const int32_t repeat, const int32_t elemsInOneRepeat, const int32_t dstBlkStride, const int32_t srcBlkStride,
    const int32_t dstRepStride, const int32_t srcRepStride, const int32_t dataSize1, const int32_t dataSize2)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize1);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize2);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize1 * sizeof(T));
    LocalTensor<T> inputLocal = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize2 * sizeof(T));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    RepeatReduceSum(outputLocal, inputLocal, repeat, elemsInOneRepeat, dstBlkStride, srcBlkStride, dstRepStride,
        srcRepStride);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    PipeBarrier<PIPE_ALL>();
}

struct RepeatReduceSumTestParams {
    const int32_t repeatIn;
    const int32_t elemsInOneRepeatIn;
    const int32_t dstBlkStrideIn;
    const int32_t srcBlkStrideIn;
    const int32_t dstRepStrideIn;
    const int32_t srcRepStrideIn;
    void (*CalFunc)(uint8_t*, uint8_t*, const int32_t, const int32_t, const int32_t, const int32_t, const int32_t,
        const int32_t, const int32_t, const int32_t);
    const int32_t dataSize1;
    const int32_t dataSize2;
    const int8_t typeByte;
    const bool expectRes;
};

class RepeatReduceSumTestsuite : public testing::Test, public testing::WithParamInterface<RepeatReduceSumTestParams> {
protected:
    void SetUp() { AscendC::SetGCoreType(2); }
    void TearDown()
    {
        AscendC::SetGCoreType(0);
        GlobalMockObject::verify();
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_REPEATREDUCESUM, RepeatReduceSumTestsuite,
    ::testing::Values(
    // 1 repeatTimes
    RepeatReduceSumTestParams{1, 64, 1, 1, 8, 8, MainRepeatReduceSum<half>,  64, 1,   2, true},   // src test
    // 2 repeatTimes
    RepeatReduceSumTestParams{ 2, 32, 1, 1, 1, 8, MainRepeatReduceSum<float>, 96, 2, 4, true}, // src + dst test
    RepeatReduceSumTestParams{ 2, 32, 1, 1, 8, 16, MainRepeatReduceSum<float>, 160, 9, 4, true}, // src + dst test
    RepeatReduceSumTestParams{ 2, 32, 1, 1, 8, 8, MainRepeatReduceSum<float>, 96, 8, 4, false}  // dst false
));

TEST_P(RepeatReduceSumTestsuite, RepeatReduceSumTestCase)
{
    auto param = GetParam();
    uint8_t dstGm[param.dataSize2 * param.typeByte] = {0};
    uint8_t srcGm[param.dataSize1 * param.typeByte] = {0};
    if (!param.expectRes) {
        MOCKER(raise, int(*)(int)).times(1).will(returnValue(0));
    }
    param.CalFunc(dstGm, srcGm, param.repeatIn, param.elemsInOneRepeatIn, param.dstBlkStrideIn, param.srcBlkStrideIn,
        param.dstRepStrideIn, param.srcRepStrideIn, param.dataSize1, param.dataSize2);

    for (int i = 1; i < param.dataSize2; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
    if (param.typeByte == B32_BYTE_SIZE) {
        float valRes = 0;
        EXPECT_EQ(GetAccVal<float>(), valRes);
    } else {
        half valRes = 0;
        EXPECT_EQ(GetAccVal<half>(), valRes);
    }
}