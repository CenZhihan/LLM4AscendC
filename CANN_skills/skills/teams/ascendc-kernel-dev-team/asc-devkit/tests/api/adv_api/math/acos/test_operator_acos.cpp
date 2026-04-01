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
    LEVEL0_COUNT_MODE,
    FIXED_BLOCK_MODE,
    EXTRA_BUFFER_MODE,
};

class TEST_ACOS : public testing::Test {
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
void MainVecAcosDemo(__gm__ uint8_t *__restrict__ dstGm, __gm__ uint8_t *__restrict__ srcGm,
    uint32_t dataSize, TestMode testMode)
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

    if (testMode == LEVEL0_COUNT_MODE) {
        Acos<T>(outputLocal, inputLocal, dataSize);
    }  else if (testMode == FIXED_BLOCK_MODE) {
        Acos<T, false>(outputLocal, inputLocal);
    } else if (testMode == EXTRA_BUFFER_MODE) {
        TBuf<TPosition::VECCALC> tbuf3;
        if (sizeof(T) == sizeof(float)) {
            tpipe.InitBuffer(tbuf3, dataSize * 2 * sizeof(float));
        } else {
            tpipe.InitBuffer(tbuf3, dataSize * 6 * sizeof(half));
        }
        LocalTensor<uint8_t> tmpLocal = tbuf3.Get<uint8_t>();
        Acos<T, false>(outputLocal, inputLocal, tmpLocal);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    PipeBarrier<PIPE_ALL>();
}
#define VEC_ACOS_LEVEL2_TESTCASE(DATA_TYPE, TEST_MODE)                                    \
    TEST_F(TEST_ACOS, Acos##DATA_TYPE##TEST_MODE##Case)                                   \
    {                                                                                     \
        uint32_t dataSize = 1024;                                                         \
        uint8_t inputGm[dataSize * sizeof(DATA_TYPE)] = {0};                                 \
        uint8_t outputGm[dataSize * sizeof(DATA_TYPE)] = {0};                                 \
                                                                                          \
        MainVecAcosDemo<DATA_TYPE>(outputGm, inputGm, dataSize, TEST_MODE); \
                                                                                          \
        for (uint32_t i = 0; i < dataSize; i++) {                                        \
            EXPECT_EQ(outputGm[i], 0x00);                                                \
        }                                                                                 \
    }

VEC_ACOS_LEVEL2_TESTCASE(float, LEVEL0_COUNT_MODE);
VEC_ACOS_LEVEL2_TESTCASE(half, LEVEL0_COUNT_MODE);
VEC_ACOS_LEVEL2_TESTCASE(float, FIXED_BLOCK_MODE);
VEC_ACOS_LEVEL2_TESTCASE(half, FIXED_BLOCK_MODE);
VEC_ACOS_LEVEL2_TESTCASE(float, EXTRA_BUFFER_MODE);
VEC_ACOS_LEVEL2_TESTCASE(half, EXTRA_BUFFER_MODE);