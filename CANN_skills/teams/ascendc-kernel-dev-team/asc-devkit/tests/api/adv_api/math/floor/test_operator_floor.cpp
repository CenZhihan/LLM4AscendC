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
    COUNT,
    COUNT_TMPBUF
};

class FloorTest : public testing::Test {
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

template <typename T>
void MainFloorDemo(
    __gm__ uint8_t *__restrict__ dstGm, __gm__ uint8_t *__restrict__ srcGm, uint32_t dataSize, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    DataCopy(inputLocal, inputGlobal, dataSize);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    constexpr int32_t TMP_BUFFER_COEF = 2;
    if (testMode == COUNT) {
        Floor<T>(outputLocal, inputLocal, dataSize);
    } else if (testMode == COUNT_TMPBUF) {
        TBuf<TPosition::VECCALC> tmpBuf;
        uint32_t tmpBufferSize =
            std::max(ONE_REPEAT_BYTE_SIZE * sizeof(float), TMP_BUFFER_COEF * dataSize * sizeof(float));
        tpipe.InitBuffer(tmpBuf, tmpBufferSize);
        LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();
        Floor<T>(outputLocal, inputLocal, tmpLocal, dataSize);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    PipeBarrier<PIPE_ALL>();
}
#define VEC_FLOOR_LEVEL2_TESTCASE(TYPE, testMode, SIZE)                       \
    TEST_F(FloorTest, Math_Floor_##TYPE##_##testMode##_##SIZE)                \
    {                                                                         \
        uint8_t inputGm[(SIZE) * sizeof(TYPE)];                               \
        uint8_t outputGm[(SIZE) * sizeof(TYPE)];                              \
        MainFloorDemo<TYPE>(outputGm, inputGm, (SIZE), testMode);             \
        for (uint32_t i = 0; i < (SIZE); i++) {                               \
            EXPECT_EQ(outputGm[i], 0x00);                                     \
        }                                                                     \
    }
VEC_FLOOR_LEVEL2_TESTCASE(float, COUNT, 256);
VEC_FLOOR_LEVEL2_TESTCASE(float, COUNT_TMPBUF, 256);

VEC_FLOOR_LEVEL2_TESTCASE(half, COUNT, 256);
VEC_FLOOR_LEVEL2_TESTCASE(half, COUNT_TMPBUF, 256);
