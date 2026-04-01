/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file test_operator_asinh.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;

enum TestMode {
    NBUFFER_NCOUNT_MODE,
    NBUFFER_COUNT_MODE,
    BUFFER_NCOUNT_MODE,
    BUFFER_COUNT_MODE
};

class TEST_ASINH : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(2);
    }
    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
    
template <typename T>
void main_vec_Asinh_level2_demo(__gm__ uint8_t *__restrict__ dstGm, __gm__ uint8_t *__restrict__ srcGm, uint32_t dataSize, TestMode testMode) {
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

    switch (testMode) {
        case NBUFFER_NCOUNT_MODE:
            Asinh<T>(outputLocal, inputLocal);
        case NBUFFER_COUNT_MODE:
            Asinh<T>(outputLocal, inputLocal, dataSize);
        case BUFFER_NCOUNT_MODE:
            TBuf<TPosition::VECCALC> tbuf3;
            tpipe.InitBuffer(tbuf3, dataSize * ASINH_HALF_CALC_PROC);
            LocalTensor<uint8_t> tmpLocal = tbuf3.Get<uint8_t>();
            Asinh<T>(outputLocal, inputLocal, tmpLocal);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    PipeBarrier<PIPE_ALL>();
}

#define VEC_ASINH_LEVEL2_TESTCASE(DATA_TYPE, testMode)                                    \
TEST_F(TEST_ASINH, Asinh##DATA_TYPE##testMode##Case)                                   \
{                                                                                     \
    uint32_t dataSize = 16384;                                                         \
    uint8_t input0Gm[dataSize * sizeof(DATA_TYPE)];                                 \
    uint8_t outputGm[dataSize * sizeof(DATA_TYPE)];                                 \
                                                                                        \
    main_vec_Asinh_level2_demo<DATA_TYPE>(outputGm, input0Gm, dataSize, testMode); \
                                                                                        \
    for (uint32_t i = 0; i < dataSize; i++) {                                        \
        EXPECT_EQ(outputGm[i], 0x00);                                                \
    }                                                                                 \
}
};

VEC_ASINH_LEVEL2_TESTCASE(float, NBUFFER_NCOUNT_MODE);
VEC_ASINH_LEVEL2_TESTCASE(half, NBUFFER_NCOUNT_MODE);
VEC_ASINH_LEVEL2_TESTCASE(float, NBUFFER_COUNT_MODE);
VEC_ASINH_LEVEL2_TESTCASE(half, NBUFFER_COUNT_MODE);
VEC_ASINH_LEVEL2_TESTCASE(float, BUFFER_NCOUNT_MODE);
VEC_ASINH_LEVEL2_TESTCASE(half, BUFFER_NCOUNT_MODE);