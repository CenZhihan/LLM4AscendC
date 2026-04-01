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

using namespace AscendC;

enum TestMode {
    CAL_MODE,
    BUF_CAL_MODE
};

class TEST_FMA : public testing::Test {
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
void MainVecFmaDemo(__gm__ uint8_t *__restrict__ dstGm, __gm__ uint8_t *__restrict__ src0Gm,
    __gm__ uint8_t *__restrict__ src1Gm, __gm__ uint8_t *__restrict__ src2Gm, uint32_t dataSize, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<T> input1Global;
    GlobalTensor<T> input2Global;
    GlobalTensor<T> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src0Gm), dataSize);
    input1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src1Gm), dataSize);
    input2Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src2Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> input0Local = tbuf1.Get<T>();   
    
    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<T> input1Local = tbuf2.Get<T>();

    TBuf<TPosition::VECCALC> tbuf3;
    tpipe.InitBuffer(tbuf3, dataSize * sizeof(T));
    LocalTensor<T> input2Local = tbuf3.Get<T>();


    TBuf<TPosition::VECCALC> tbuf4;
    tpipe.InitBuffer(tbuf4, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf4.Get<T>();

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);
    DataCopy(input2Local, input2Global, dataSize);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if (testMode == CAL_MODE) {
        Fma(outputLocal, input0Local, input1Local, input2Local, dataSize);
    } else {
        TBuf<TPosition::VECCALC> tbuf4;
        if (sizeof(T) == sizeof(float)) {
            tpipe.InitBuffer(tbuf4, dataSize * sizeof(float));
        } else {
            tpipe.InitBuffer(tbuf4, dataSize * sizeof(float) * 4);
        }
        LocalTensor<uint8_t> tmpLocal = tbuf4.Get<uint8_t>();
        Fma(outputLocal, input0Local, input1Local, input2Local, tmpLocal, dataSize);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    PipeBarrier<PIPE_ALL>();
}
#define VEC_FMA_TESTCASE(DATA_TYPE, TEST_MODE, DATA_SIZE)                         \
    TEST_F(TEST_FMA, Fma##DATA_TYPE##TEST_MODE##DATA_SIZE##Case)                       \
    {                                                                                      \
        uint32_t dataSize = DATA_SIZE;                                                    \
        uint8_t input0Gm[dataSize * sizeof(DATA_TYPE)] = {0};                                  \
        uint8_t input1Gm[dataSize * sizeof(DATA_TYPE)] = {0};                                   \
        uint8_t input2Gm[dataSize * sizeof(DATA_TYPE)] = {0};                                   \
        uint8_t outputGm[dataSize * sizeof(DATA_TYPE)] = {0};                                  \
                                                                                           \
        MainVecFmaDemo<DATA_TYPE>(outputGm, input0Gm, input1Gm, input2Gm, dataSize, TEST_MODE); \
                                                                                           \
        for (uint32_t i = 0; i < dataSize; i++) {                                         \
            EXPECT_EQ(outputGm[i], 0x00);                                                 \
        }                                                                                  \
    }
VEC_FMA_TESTCASE(float, CAL_MODE, 256);
VEC_FMA_TESTCASE(float, BUF_CAL_MODE, 256);

VEC_FMA_TESTCASE(half, CAL_MODE, 256);
VEC_FMA_TESTCASE(half, BUF_CAL_MODE, 256);
