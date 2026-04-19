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
void VecWholeReduceSum(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm,
    __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<T> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    LocalTensor<T> input0Local;
    TBuffAddr tbuf0;
    tbuf0.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    input0Local.SetAddr(tbuf0);
    input0Local.InitBuffer(0, dataSize);

    LocalTensor<T> outputLocal;
    TBuffAddr tbuf2;
    tbuf2.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    outputLocal.SetAddr(tbuf2);
    outputLocal.InitBuffer(dataSize * sizeof(T) * 2, dataSize);

    DataCopy(input0Local, input0Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    uint64_t mask[2] = {0xffffffff, 0xffffffff};
    uint64_t counterMask[2] = {144, 0};
    uint8_t repeatTimes = dataSize / (256 / sizeof(T));
    WholeReduceSum<T>(outputLocal, input0Local, mask, repeatTimes, 1, 1, 8);

    AscendC::SetMaskCount();
    AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, 144);
    WholeReduceSum<T, false>(outputLocal, input0Local, AscendC::MASK_PLACEHOLDER_LIST, repeatTimes, 1, 1, 8);
    AscendC::ResetMask();
    WholeReduceSum<T>(outputLocal, input0Local, counterMask, repeatTimes, 1, 1, 8);
    AscendC::SetMaskNorm();

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

struct ReduceTestParams {
    int32_t dataSize;
    int32_t databitSize;
    void (*cal_func)(uint8_t*, uint8_t*, int32_t);
};

class ReduceSimpleTestsuite : public testing::Test, public testing::WithParamInterface<ReduceTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(ReduceSimpleTestCase, ReduceSimpleTestsuite,
    ::testing::Values(ReduceTestParams{ 256, 2, VecWholeReduceSum<half> }));

TEST_P(ReduceSimpleTestsuite, ReduceSimpleTestCase)
{
    TPipe tpipe;
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.databitSize] = {0};
    uint8_t dstGm[param.dataSize * param.databitSize] = {0};

    param.cal_func(dstGm, srcGm, param.dataSize);
    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

template <typename T>
__global__ __aicore__ void MainRepeatReduceSum(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm,
    const int32_t repeat, const int32_t elemsInOneRepeat, const int32_t dstBlkStride, const int32_t srcBlkStride,
    const int32_t dstRepStride, const int32_t srcRepStride, const int32_t dataSize1, const int32_t dataSize2)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(srcGm), dataSize1);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(dstGm), dataSize2);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize1 * sizeof(PrimT<T>));
    LocalTensor<T> inputLocal = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize2 * sizeof(PrimT<T>));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    DataCopy(inputLocal, inputGlobal, dataSize1);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    RepeatReduceSum(outputLocal, inputLocal, repeat, elemsInOneRepeat, dstBlkStride, srcBlkStride, dstRepStride,
        srcRepStride);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopy(outputGlobal, outputLocal, dataSize2);

    PipeBarrier<PIPE_ALL>();
}

struct RepeatReduceSumTestParams {
    const int32_t repeatIn;
    const int32_t elemsInOneRepeatIn;
    const int32_t dstBlkStrideIn;
    const int32_t srcBlkStrideIn;
    const int32_t dstRepStrideIn;
    const int32_t srcRepStrideIn;
    void (*cal_func)(uint8_t*, uint8_t*, const int32_t, const int32_t, const int32_t, const int32_t, const int32_t,
        const int32_t, const int32_t, const int32_t);
    const int32_t dataSize1;
    const int32_t dataSize2;
    const int8_t typeByte;
};

class RepeatReduceSumTestsuite : public testing::Test, public testing::WithParamInterface<RepeatReduceSumTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_REPEATREDUCESUM, RepeatReduceSumTestsuite,
    ::testing::Values(RepeatReduceSumTestParams{ 1, 64, 1, 1, 8, 8, MainRepeatReduceSum<half>, 128, 16, 2 },
    RepeatReduceSumTestParams{ 2, 32, 1, 1, 8, 8, MainRepeatReduceSum<float>, 128, 16, 4 },
    // TensorTrait Case
    RepeatReduceSumTestParams{ 1, 64, 1, 1, 8, 8, MainRepeatReduceSum<TensorTrait<half>>, 128, 16, 2 },
    RepeatReduceSumTestParams{ 2, 32, 1, 1, 8, 8, MainRepeatReduceSum<TensorTrait<float>>, 128, 16, 4 }));

TEST_P(RepeatReduceSumTestsuite, RepeatReduceSumTestCase)
{
    auto param = GetParam();
    uint8_t dstGm[param.dataSize2 * param.typeByte] = {0};
    uint8_t srcGm[param.dataSize1 * param.typeByte] = {0};

    param.cal_func(dstGm, srcGm, param.repeatIn, param.elemsInOneRepeatIn, param.dstBlkStrideIn, param.srcBlkStrideIn,
        param.dstRepStrideIn, param.srcRepStrideIn, param.dataSize1, param.dataSize2);

    for (int i = 1; i < param.dataSize2; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

template <typename T>
__global__ __aicore__ void MainAllReduceSimple(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm,
    __gm__ int32_t srcDataSize, __gm__ int32_t dstDataSize, __gm__ int32_t level)
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

    TBuf<TPosition::VECIN> tbuf1;
    tpipe.InitBuffer(tbuf1, srcDataSize * sizeof(PrimT<T>));
    LocalTensor<T> inputLocal = tbuf1.Get<T>();

    TBuf<TPosition::VECOUT> tbuf2;
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
        case 2:
            ReduceMax<T>(outputLocal, inputLocal, workLocal, srcDataSize, false);
            ReduceMax<T>(outputLocal, inputLocal, workLocal, srcDataSize, true);
            ReduceSum<T>(outputLocal, inputLocal, workLocal, srcDataSize);
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

struct ReduceMergeTestParams {
    void (*calFunc)(uint8_t*, uint8_t*, int32_t, int32_t, int32_t);
    int32_t srcDataSize;
    int32_t dstDataSize;
    int32_t dtypeSize;
    int32_t level;
};

class ReduceMergeSimpleTestsuite : public testing::Test, public testing::WithParamInterface<ReduceMergeTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_REDUCE_SIMPLE, ReduceMergeSimpleTestsuite,
    ::testing::Values(
    ReduceMergeTestParams{ MainAllReduceSimple<float>, 64, 16, 4, 2 },
    ReduceMergeTestParams{ MainAllReduceSimple<float>, 512, 16, 4, 2 },
    ReduceMergeTestParams{ MainAllReduceSimple<float>, 8192, 16, 4, 2 },
    ReduceMergeTestParams{ MainAllReduceSimple<half>, 512, 16, 2, 2 },
    // TensorTrait Case
    ReduceMergeTestParams{ MainAllReduceSimple<TensorTrait<float>>, 8192, 16, 4, 2 },
    ReduceMergeTestParams{ MainAllReduceSimple<TensorTrait<half>>, 512, 16, 2, 2 }
    ));

TEST_P(ReduceMergeSimpleTestsuite, ReduceMergeSimpleTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.srcDataSize * param.dtypeSize];
    uint8_t dstGm[param.dstDataSize * param.dtypeSize];

    param.calFunc(srcGm, dstGm, param.srcDataSize, param.dstDataSize, param.level);
    for (int32_t i = 0; i < param.dstDataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
