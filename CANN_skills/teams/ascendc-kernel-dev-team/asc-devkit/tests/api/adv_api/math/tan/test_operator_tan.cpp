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
    NORMAL_MODE,
    CAL_MODE,
    BUF_MODE,
    BUF_CAL_MODE
};

class TEST_TAN : public testing::Test {
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
void MainVecTanLevel2Demo(__gm__ uint8_t* __restrict__ dstGm,
    __gm__ uint8_t* __restrict__ srcGm, uint32_t dataSize, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    DataCopy(inputLocal, inputGlobal, dataSize);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if (testMode == NORMAL_MODE) {
        Tan<T>(outputLocal, inputLocal);
    } else if (testMode == CAL_MODE) {
        Tan<T>(outputLocal, inputLocal, dataSize);
    } else {
        TBuf<TPosition::VECCALC> tbuf3;
        if (sizeof(T) == sizeof(float)) {
            tpipe.InitBuffer(tbuf3, dataSize * 4 * sizeof(float));
        } else {
            tpipe.InitBuffer(tbuf3, dataSize * 10 * sizeof(half));
        }
        LocalTensor<uint8_t> tmpLocal = tbuf3.Get<uint8_t>();
        if (testMode == BUF_MODE) {
            Tan<T>(outputLocal, inputLocal, tmpLocal);
        }
        else if (testMode == BUF_CAL_MODE){
            Tan<T>(outputLocal, inputLocal, tmpLocal, dataSize);
        }
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    PipeBarrier<PIPE_ALL>();
}
#define VEC_TAN_LEVEL2_TESTCASE(DATA_TYPE, testMode)                            \
    TEST_F(TEST_TAN, Tan##DATA_TYPE##testMode##Case)                            \
    {                                                                           \
        uint32_t dataSize = 256;                                                \
        uint8_t inputGm[dataSize * sizeof(DATA_TYPE)];                          \
        uint8_t outputGm[dataSize * sizeof(DATA_TYPE)];                         \
                                                                                \
        MainVecTanLevel2Demo<DATA_TYPE>(outputGm, inputGm, dataSize, testMode); \
                                                                                \
        for (uint32_t i = 0; i < dataSize; i++) {                               \
            EXPECT_EQ(outputGm[i], 0x00);                                       \
        }                                                                       \
    }
VEC_TAN_LEVEL2_TESTCASE(half, NORMAL_MODE);
VEC_TAN_LEVEL2_TESTCASE(half, CAL_MODE);
VEC_TAN_LEVEL2_TESTCASE(half, BUF_MODE);
VEC_TAN_LEVEL2_TESTCASE(half, BUF_CAL_MODE);
VEC_TAN_LEVEL2_TESTCASE(float, NORMAL_MODE);
VEC_TAN_LEVEL2_TESTCASE(float, CAL_MODE);
VEC_TAN_LEVEL2_TESTCASE(float, BUF_MODE);
VEC_TAN_LEVEL2_TESTCASE(float, BUF_CAL_MODE);