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

class TEST_ATANH : public testing::Test {
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
void main_vec_atanh_level2_demo(__gm__ uint8_t *__restrict__ dstGm, __gm__ uint8_t *__restrict__ src0Gm,
    uint32_t dataSize, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<T> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src0Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> input0Local = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    DataCopy(input0Local, input0Global, dataSize);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if (testMode == LEVEL0_COUNT_MODE) {
        Atanh<T>(outputLocal, input0Local);
    } else if (testMode == FIXED_BLOCK_MODE) {
        Atanh<T, false>(outputLocal, input0Local);
    } else if (testMode == EXTRA_BUFFER_MODE) {
        TBuf<TPosition::VECCALC> tbuf3;
        if (sizeof(T) == sizeof(float)) {
            tpipe.InitBuffer(tbuf3, dataSize * 1);
        } else {
            tpipe.InitBuffer(tbuf3, dataSize * 2);
        }
        LocalTensor<uint8_t> tmpLocal = tbuf3.Get<uint8_t>();

        Atanh<T>(outputLocal, input0Local, tmpLocal);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    PipeBarrier<PIPE_ALL>();
}

#define VEC_ATANH_LEVEL2_TESTCASE(DATA_TYPE, testMode)                                 \
    TEST_F(TEST_ATANH, Atanh##DATA_TYPE##testMode##Case)                                \
    {                                                                                     \
        uint32_t dataSize = 16384;                                                         \
        uint8_t input0Gm[dataSize * sizeof(DATA_TYPE)];                                 \
        uint8_t outputGm[dataSize * sizeof(DATA_TYPE)];                                 \
                                                                                          \
        main_vec_atanh_level2_demo<DATA_TYPE>(outputGm, input0Gm, dataSize, testMode); \
                                                                                          \
        for (uint32_t i = 0; i < dataSize; i++) {                                        \
            EXPECT_EQ(outputGm[i], 0x00);                                                \
        }                                                                                 \
    }
VEC_ATANH_LEVEL2_TESTCASE(float, LEVEL0_COUNT_MODE);
VEC_ATANH_LEVEL2_TESTCASE(half, LEVEL0_COUNT_MODE);
VEC_ATANH_LEVEL2_TESTCASE(float, FIXED_BLOCK_MODE);
VEC_ATANH_LEVEL2_TESTCASE(half, FIXED_BLOCK_MODE);
VEC_ATANH_LEVEL2_TESTCASE(float, EXTRA_BUFFER_MODE);
VEC_ATANH_LEVEL2_TESTCASE(half, EXTRA_BUFFER_MODE);
