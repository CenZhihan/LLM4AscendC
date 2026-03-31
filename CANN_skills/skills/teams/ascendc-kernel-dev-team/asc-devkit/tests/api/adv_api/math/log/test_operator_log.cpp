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
    LOG_MODE,
    LOG2_MODE,
    LOG10_MODE,
};

class TestLog : public testing::Test {
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
void MainVecSelectCase(__gm__ uint8_t* __restrict__ dstGm,
    __gm__ uint8_t* __restrict__ srcGm0, uint32_t dataSize,TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal0;
    GlobalTensor<T> outputGlobal;
    inputGlobal0.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm0), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    DataCopy(inputLocal, inputGlobal0, dataSize);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if (testMode == LOG_MODE) {
        Log(outputLocal, inputLocal);
    } else if (testMode == LOG2_MODE) {
        Log2(outputLocal, inputLocal);
    } else if (testMode == LOG10_MODE) {
        Log10(outputLocal, inputLocal);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    PipeBarrier<PIPE_ALL>();
}
#define VEC_LOG_LEVEL2_TESTCASE(DATA_TYPE, TEST_MODE)                                              \
    TEST_F(TestLog, Log##DATA_TYPE##TEST_MODE##Case)                                           \
    {                                                                                                           \
        uint32_t dataSize = 256;                                                                               \
        uint32_t sel_mask_size = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));                         \
        uint8_t inputGm[dataSize * sizeof(DATA_TYPE)];                                                       \
        uint8_t outputGm[dataSize * sizeof(DATA_TYPE)];                                                       \
                                                                                                                \
        MainVecSelectCase<DATA_TYPE>(outputGm, inputGm, dataSize, TEST_MODE);                     \
                                                                                                                \
        for (uint32_t i = 0; i < dataSize; i++) {                                                              \
            EXPECT_EQ(outputGm[i], 0x00);                                                                      \
        }                                                                                                       \
    }
VEC_LOG_LEVEL2_TESTCASE(float, LOG_MODE);
VEC_LOG_LEVEL2_TESTCASE(half, LOG_MODE);
VEC_LOG_LEVEL2_TESTCASE(float, LOG2_MODE);
VEC_LOG_LEVEL2_TESTCASE(half, LOG2_MODE);
VEC_LOG_LEVEL2_TESTCASE(float, LOG10_MODE);
VEC_LOG_LEVEL2_TESTCASE(half, LOG10_MODE);
