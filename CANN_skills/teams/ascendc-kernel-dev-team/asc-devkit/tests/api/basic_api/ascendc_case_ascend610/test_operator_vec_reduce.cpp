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
#include "kernel_log.h"
static uint8_t g_testRes = 1;      // 全局变量记录运行结果, 如果进入ASCENDC_ASSERT报错，会被置为0
// 重定义ASCENDC_ASSERT，不Abort，仅修改全局变量通知进入报错分支
#undef ASCENDC_ASSERT
#define ASCENDC_ASSERT(cond, behavior) \
    do {                               \
        if (!(cond)) {                 \
            g_testRes = 0;              \
            behavior;                  \
        }                              \
    } while (0)

#undef ASCENDC_REPORT_CHECK_ERROR
#define ASCENDC_REPORT_CHECK_ERROR(apiMsg, funcType)   \
    do {                                                \
        g_testRes = 0;                                   \
    } while (0)

#include "kernel_utils.h"
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;

template <typename T>
void MainVecReduceDemo(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm,
    __gm__ int32_t srcDataSize, __gm__ int32_t dstDataSize)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dstDataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcDataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dstDataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    AscendCUtils::SetMask<uint8_t>(256);
    DataCopy(inputLocal, inputGlobal, srcDataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(256);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    BlockReduceMax<T>(outputLocal, inputLocal, 2, 64, 8, 1, 8);
    BlockReduceMin<T>(outputLocal, inputLocal, 1, 64, 8, 1, 8);
    BlockReduceSum<T>(outputLocal, inputLocal, 1, 64, 8, 1, 8);
    PairReduceSum<T>(outputLocal, inputLocal, 2, 64, 8, 1, 8);

    SetMaskCount();
    set_vector_mask(0, 64);
    BlockReduceSumIntrinsicsImpl((__ubuf__ T*)outputLocal.GetPhyAddr(), (__ubuf__ T*)inputLocal.GetPhyAddr(), 1, 1, 1,
        8);
    BlockReduceMinIntrinsicsImpl((__ubuf__ T*)outputLocal.GetPhyAddr(), (__ubuf__ T*)inputLocal.GetPhyAddr(), 1, 1, 1,
        8);
    BlockReduceMaxIntrinsicsImpl((__ubuf__ T*)outputLocal.GetPhyAddr(), (__ubuf__ T*)inputLocal.GetPhyAddr(), 1, 1, 1,
        8);
    PairReduceSumIntrinsicsImpl((__ubuf__ T*)outputLocal.GetPhyAddr(), (__ubuf__ T*)inputLocal.GetPhyAddr(), 1, 1, 1,
        8);
    SetMaskNorm();
    AscendCUtils::ResetMask();

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    pipe_barrier(PIPE_ALL);
}

class TEST_VEC_REDUCE : public testing::Test {
protected:
    void SetUp() {}

    void TearDown() {}
};

TEST_F(TEST_VEC_REDUCE, OperatorVecReduceFloatCase)
{
    const int32_t srcDataSize = 128;
    const int32_t dstDataSize = 64;

    uint8_t srcGm[srcDataSize * sizeof(float)];
    uint8_t dstGm[dstDataSize * sizeof(float)];

    MainVecReduceDemo<float>(srcGm, dstGm, srcDataSize, dstDataSize);

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

    MainVecReduceDemo<half>(srcGm, dstGm, srcDataSize, dstDataSize);

    for (int32_t i = 0; i < dstDataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

template <typename T>
__global__ __aicore__ void MainAllReduceSimple(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm,
    __gm__ int32_t srcDataSize, __gm__ int32_t dstDataSize, __gm__ int32_t level)
{
    TPipe tpipe;
    int32_t mask = 64;
    uint64_t masks[2] { FULL_MASK, 0 };
    if (sizeof(T) == sizeof(half)) {
        mask = 128;
        masks[0] = FULL_MASK;
        masks[1] = FULL_MASK;
    }
    int32_t repStride = 8;
    int32_t repeat = srcDataSize / mask;

    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dstDataSize);

    // [0 -- srcDataSize*sizeof(T)]
    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcDataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    // [srcDataSize*sizeof(T) -- srcDataSize*sizeof(T) + dstDataSize*sizeof(T)]
    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dstDataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    // [(srcDataSize + dstDataSize) * sizeof(T) -- (srcDataSize + dstDataSize) * sizeof(T) + repeat * 3]
    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, AlignUp(repeat * 3 * sizeof(T), ONE_BLK_SIZE));
    LocalTensor<T> workLocal = tbuf2.Get<T>();

    DataCopy(inputLocal, inputGlobal, srcDataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

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
            ReduceMax<T>(outputLocal, inputLocal, workLocal, srcDataSize, false);
            ReduceMax<T>(outputLocal, inputLocal, workLocal, srcDataSize, true);
            ReduceMin<T>(outputLocal, inputLocal, workLocal, srcDataSize, false);
            ReduceMin<T>(outputLocal, inputLocal, workLocal, srcDataSize, true);
            ReduceSum<T>(outputLocal, inputLocal, workLocal, srcDataSize);
            break;
        default:
            break;
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    pipe_barrier(PIPE_ALL);
}

template <typename T>
__global__ __aicore__ void MainWholeReduceSimple(__gm__ uint8_t* __restrict__ srcGm,
    __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t srcDataSize, __gm__ int32_t dstDataSize,
    __gm__ int32_t level)
{
    TPipe tpipe;
    int32_t mask = 64;
    uint64_t masks[2] { FULL_MASK, 0 };
    if (sizeof(T) == sizeof(half)) {
        mask = 128;
        masks[0] = FULL_MASK;
        masks[1] = FULL_MASK;
    }
    int32_t repStride = 8;
    int32_t repeat = srcDataSize / mask;

    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dstDataSize);

    // [0 -- srcDataSize*sizeof(T)]
    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcDataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    // [srcDataSize*sizeof(T) -- srcDataSize*sizeof(T) + dstDataSize*sizeof(T)]
    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dstDataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    // [(srcDataSize + dstDataSize) * sizeof(T) -- (srcDataSize + dstDataSize) * sizeof(T) + repeat * 3]
    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, AlignUp(repeat * 3 * sizeof(T), ONE_BLK_SIZE));
    LocalTensor<T> workLocal = tbuf2.Get<T>();

    DataCopy(inputLocal, inputGlobal, srcDataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    WholeReduceMax<T>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);
    WholeReduceMax<T>(outputLocal, inputLocal, mask, 3, 8, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);
    WholeReduceMax<T>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_INDEX_VALUE);
    WholeReduceSum<T>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride);
    WholeReduceMin<T>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);
    WholeReduceMin<T>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_INDEX_VALUE);

    WholeReduceMax<T>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);
    WholeReduceMax<T>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_INDEX_VALUE);
    WholeReduceSum<T>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride);
    WholeReduceMin<T>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);
    WholeReduceMin<T>(outputLocal, inputLocal, masks, repeat, 1, 1, repStride, ReduceOrder::ORDER_INDEX_VALUE);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    pipe_barrier(PIPE_ALL);
}
struct ReduceTestParams {
    void (*calFunc)(uint8_t*, uint8_t*, int32_t, int32_t, int32_t);
    int32_t srcDataSize;
    int32_t dstDataSize;
    int32_t dtypeSize;
    int32_t level;
};

class ReduceSimpleTestsuite : public testing::Test, public testing::WithParamInterface<ReduceTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_REDUCE_SIMPLE, ReduceSimpleTestsuite,
    ::testing::Values(ReduceTestParams { MainAllReduceSimple<half>, 128, 16, 2, 0 },
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002) // dav-m200
    ReduceTestParams { MainAllReduceSimple<float>, 64, 16, 4, 0 },
    ReduceTestParams { MainAllReduceSimple<float>, 64, 16, 4, 2 },
    ReduceTestParams { MainAllReduceSimple<float>, 16648, 16, 4, 0 },
    ReduceTestParams { MainAllReduceSimple<float>, 16648, 16, 4, 2 },

    ReduceTestParams { MainWholeReduceSimple<float>, 64, 16, 4, 0 },
    ReduceTestParams { MainWholeReduceSimple<float>, 256, 16, 4, 0 },
#endif
    ReduceTestParams { MainAllReduceSimple<half>, 128, 16, 2, 0 },
    ReduceTestParams { MainAllReduceSimple<half>, 128, 16, 2, 2 },
    ReduceTestParams { MainAllReduceSimple<half>, 32656, 16, 2, 0 },
    ReduceTestParams { MainAllReduceSimple<half>, 32656, 16, 2, 2 },

    ReduceTestParams { MainWholeReduceSimple<half>, 128, 16, 2, 0 }));

TEST_P(ReduceSimpleTestsuite, ReduceSimpleTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.srcDataSize * param.dtypeSize];
    uint8_t dstGm[param.dstDataSize * param.dtypeSize];

    param.calFunc(srcGm, dstGm, param.srcDataSize, param.dstDataSize, param.level);
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

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcDataSize * sizeof(float));
    LocalTensor<float> inputLocal = tbuf.Get<float>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dstDataSize * sizeof(float));
    LocalTensor<float> outputLocal = tbuf1.Get<float>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, AlignUp(repeat * 3 * sizeof(float), ONE_BLK_SIZE));
    LocalTensor<float> workLocal = tbuf2.Get<float>();

    DataCopy(inputLocal, inputGlobal, srcDataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    WholeReduceMax<float>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);

    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    uint32_t maxMinVal = 0;
    GetReduceMaxMinCount<float>(maxMinVal);
    EXPECT_EQ(maxMinVal, 0x0);

    float maxMinFp32Val = 0;
    GetReduceMaxMinCount<float>(maxMinFp32Val);
    EXPECT_EQ(maxMinFp32Val, 0x0);

    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    pipe_barrier(PIPE_ALL);
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

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcDataSize * sizeof(float));
    LocalTensor<float> inputLocal = tbuf.Get<float>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dstDataSize * sizeof(float));
    LocalTensor<float> outputLocal = tbuf1.Get<float>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, AlignUp(repeat * 3 * sizeof(float), ONE_BLK_SIZE));
    LocalTensor<float> workLocal = tbuf2.Get<float>();

    DataCopy(inputLocal, inputGlobal, srcDataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    WholeReduceMax<float>(outputLocal, inputLocal, mask, repeat, 1, 1, repStride, ReduceOrder::ORDER_VALUE_INDEX);

    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    uint32_t maxMinVal = 0;
    GetReduceRepeatMaxMinSpr<float>(maxMinVal);
    EXPECT_EQ(maxMinVal, 0x0);

    float maxMinFp32Val = 0;
    GetReduceRepeatMaxMinSpr<float>(maxMinFp32Val);
    EXPECT_EQ(maxMinFp32Val, 0x0);

    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    pipe_barrier(PIPE_ALL);
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

    TBuf<TPosition::CO2> tbuf;
    tpipe.InitBuffer(tbuf, dataSize1 * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    TBuf<TPosition::CO2> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize2 * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    DataCopy(inputLocal, inputGlobal, dataSize1);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    RepeatReduceSum(outputLocal, inputLocal, repeat, elemsInOneRepeat, dstBlkStride, srcBlkStride, dstRepStride,
        srcRepStride);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize2);

    pipe_barrier(PIPE_ALL);
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
    const int32_t datasize1;
    const int32_t datasize2;
    const int8_t typeByte;
};

class RepeatReduceSumTestsuite : public testing::Test, public testing::WithParamInterface<RepeatReduceSumTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_REPEATREDUCESUM, RepeatReduceSumTestsuite,
    ::testing::Values(RepeatReduceSumTestParams { 1, 64, 1, 1, 8, 8, MainRepeatReduceSum<half>, 128, 16, 2 },
    RepeatReduceSumTestParams { 2, 32, 1, 1, 8, 8, MainRepeatReduceSum<float>, 128, 16, 4 }));

TEST_P(RepeatReduceSumTestsuite, RepeatReduceSumTestCase)
{
    auto param = GetParam();
    uint8_t dstGm[param.datasize1 * param.typeByte];
    uint8_t srcGm[param.datasize2 * param.typeByte];

    param.cal_func(dstGm, srcGm, param.repeatIn, param.elemsInOneRepeatIn, param.dstBlkStrideIn, param.srcBlkStrideIn,
        param.dstRepStrideIn, param.srcRepStrideIn, param.datasize1, param.datasize2);

    for (int i = 1; i < param.datasize2; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}