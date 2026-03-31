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

bool CheckMaskValue0(uint64_t mask) { return mask == 0; }
bool CheckMaskValueFullMask(uint64_t mask) { return mask == 0xffffffffffffffff; }
bool CheckMaskValue9(uint64_t mask) { return mask == 0x1ff; }

template <typename T, bool isNorm>
void UpdateMask(uint32_t maskLen, uint32_t maskLow, uint32_t maskHigh)
{
    if constexpr(isNorm) {
        SetMaskNorm();
        if (maskLen == 1) {
            SetVectorMask<T, MaskMode::NORMAL>(maskLow);
        } else {
            SetVectorMask<T, MaskMode::NORMAL>(maskHigh, maskLow);
        }
    } else {
        SetMaskCount();
        if (maskLen == 1) {
            SetVectorMask<T, MaskMode::COUNTER>(maskLow);
        } else {
            SetVectorMask<T, MaskMode::COUNTER>(maskHigh, maskLow);
        }
    }
}

template <typename T, bool isSetMask, bool isNorm = true>
void MainSetVectorMaskNoCalcDemo(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm,
    T scalarValue, uint32_t dataSize, uint32_t maskLen, uint32_t maskLow, uint32_t maskHigh)
{
    UpdateMask<T, isNorm>(maskLen, maskLow, maskHigh);
}

template <typename T, bool isSetMask, bool isNorm = true>
void MainSetVectorMaskCalcDemo(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm,
    T scalarValue, uint32_t dataSize, uint32_t maskLen, uint32_t maskLow, uint32_t maskHigh)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    TBuf<TPosition::VECCALC> tbuf;
    TBuf<TPosition::VECCALC> tbuf1;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    UpdateMask<T, isNorm>(maskLen, maskLow, maskHigh);

    DataCopy(inputLocal, inputGlobal, dataSize);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_ALL);

    EXPECT_EQ(MaskSetter::Instance().GetMask(), true);
    UnaryRepeatParams params = {1, 1, 8, 8};
    Adds<T, isSetMask>(outputLocal, inputLocal, scalarValue, 20, 1, params);
    EXPECT_EQ(MaskSetter::Instance().GetMask(), true);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}


class TEST_VEC_SET_VEC_MASK : public testing::Test {
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

// 每次调用完api后，isSetMask应该被重置为true
TEST_F(TEST_VEC_SET_VEC_MASK, VecMaskTestSuite_MaskSetter_Reset)                                                       \
{                                                                                                                      \
    uint32_t maskLen = 1;                                                                                              \
    uint64_t maskLow = 73, maskHigh = 0;                                                                               \
    uint8_t srcGm[512 * sizeof(int32_t)] { 0x00 };                                                                     \
    uint8_t outputGm[512 * sizeof(int32_t)] { 0x00 };                                                                  \
    MainSetVectorMaskCalcDemo<int32_t, false, false>(srcGm, outputGm, static_cast<int32_t>(1), 512, maskLen, maskLow, maskHigh); \
    SetMaskNorm();                                                                                                     \
}