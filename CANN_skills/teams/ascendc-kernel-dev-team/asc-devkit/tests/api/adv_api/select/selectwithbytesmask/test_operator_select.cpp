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

enum TestMode {
    LEVEL2,
};

class SelectTest : public testing::Test {
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

template <typename T, typename U, bool reuse>
void MainVecSelectDemo(__gm__ uint8_t *__restrict__ dstGm,
    __gm__ uint8_t *__restrict__ selMaskGm, __gm__ uint8_t *__restrict__ srcGm, uint32_t firstAxis,
    uint32_t srcLastAxis, uint32_t maskLastAxis, bool reverse, TestMode testMode)
{
    uint32_t dataSize = firstAxis * srcLastAxis;
    uint32_t selMaskSize = firstAxis * maskLastAxis;

    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<U> selMaskGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), dataSize);
    selMaskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(selMaskGm), selMaskSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    TBuf<TPosition::VECCALC> tbuf3;
    tpipe.InitBuffer(tbuf3, selMaskSize * sizeof(U));
    LocalTensor<U> selMaskLocal = tbuf3.Get<U>();

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    TBuf<TPosition::VECCALC> tmpBuf;
    uint32_t tmpBufferSize =
        std::max(static_cast<uint32_t>(1024), (dataSize + 31) / 32 * 32 * 2 + 256);
    if constexpr (!reuse) {
        tmpBufferSize += (dataSize + 31) / 32 * 32 * sizeof(U);
    }
    tpipe.InitBuffer(tmpBuf, tmpBufferSize);
    LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();

    SelectShapeInfo info;
    info.firstAxis = firstAxis;
    info.srcLastAxis = srcLastAxis;
    info.maskLastAxis = maskLastAxis;

    if (!reverse) {
        Select<T, U, reuse>(outputLocal, inputLocal, static_cast<T>(100), selMaskLocal,
            tmpLocal, info);
    } else {
        Select<T, U, reuse>(outputLocal, static_cast<T>(100), inputLocal, selMaskLocal,
            tmpLocal, info);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    PipeBarrier<PIPE_ALL>();
}

#define VEC_SELECT_LEVEL2_TESTCASE(DATA_TYPE, MASK_TYPE, SRC_AXIS, MASK_AXIS, REVERSE, REUSE,       \
    testMode)                                                                                                      \
    TEST_F(SelectTest,                                                                             \
        Select##DATA_TYPE##MASK_TYPE##SRC_AXIS##MASK_AXIS##REVERSE##REUSE##testMode##Case)            \
    {                                                                                                               \
        uint32_t firstAxis = 8;                                                                                    \
        uint32_t srcLastAxis = SRC_AXIS;                                                                          \
        uint32_t maskLastAxis = MASK_AXIS;                                                                        \
        uint32_t dataSize = firstAxis * srcLastAxis;                                                            \
        uint32_t selMaskSize = firstAxis * maskLastAxis;                                                       \
        uint8_t input0Gm[dataSize * sizeof(DATA_TYPE)] = {0};                                                           \
        uint8_t selMaskGm[selMaskSize * sizeof(MASK_TYPE)] = {0};                                                          \
        uint8_t outputGm[dataSize * sizeof(DATA_TYPE)] = {0};                                                           \
                                                                                                                    \
        MainVecSelectDemo<DATA_TYPE, MASK_TYPE, REUSE>(outputGm, selMaskGm, input0Gm, \
            firstAxis, srcLastAxis, maskLastAxis, REVERSE, testMode);                                         \
                                                                                                                    \
        for (uint32_t i = 0; i < dataSize; i++) {                                                                  \
            EXPECT_EQ(outputGm[i], 0x00);                                                                          \
        }                                                                                                           \
    }
VEC_SELECT_LEVEL2_TESTCASE(half, bool, 32, 32, false, false, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(half, int16_t, 32, 32, false, false, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(half, int32_t, 32, 32, false, false, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(float, bool, 32, 32, false, false, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(float, int16_t, 32, 32, false, false, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(float, int32_t, 32, 32, false, false, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(half, bool, 48, 64, false, false, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(half, int16_t, 32, 32, true, false, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(float, bool, 32, 32, true, false, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(half, int32_t, 32, 64, false, false, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(float, bool, 32, 64, false, false, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(float, int16_t, 32, 64, false, false, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(float, int32_t, 32, 64, false, false, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(half, bool, 32, 64, true, false, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(half, bool, 48, 64, false, true, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(half, bool, 48, 64, true, true, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(float, bool, 48, 64, false, true, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(float, bool, 48, 64, true, true, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(half, uint16_t, 48, 64, false, true, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(float, uint16_t, 48, 64, false, true, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(half, uint32_t, 48, 64, false, true, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(float, uint32_t, 48, 64, false, true, LEVEL2);
