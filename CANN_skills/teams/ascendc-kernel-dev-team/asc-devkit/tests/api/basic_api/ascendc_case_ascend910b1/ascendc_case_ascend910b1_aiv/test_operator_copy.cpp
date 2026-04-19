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
    LEVEL0_BIT_MODE,
    LEVEL0_COUNT_MODE,
};

class TestCopy : public testing::Test {
protected:
    void SetUp() {AscendC::SetGCoreType(2);}
    void TearDown() {AscendC::SetGCoreType(0);}
};

template <typename Type>
void MainVecCopyDemo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm,
    uint8_t repeat, CopyRepeatParams repeatParams, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<Type> inputGlobal;
    GlobalTensor<Type> outputGlobal;
    int dataSize = 256 * repeat / sizeof(Type);
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ Type*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ Type*>(dstGm), dataSize);

    LocalTensor<Type> inputLocal;
    LocalTensor<Type> outputLocal;

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(Type));
    inputLocal = tbuf.Get<Type>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(Type));
    outputLocal = tbuf1.Get<Type>();

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if (testMode == TestMode::LEVEL0_COUNT_MODE) {
        uint64_t mask = 256 / sizeof(Type);
        Copy<Type, true>(outputLocal, inputLocal, mask, repeat, repeatParams);
    } else if (testMode == TestMode::LEVEL0_BIT_MODE) {
        uint64_t mask[2];
        if (sizeof(Type) == 2) {
            mask[0] = UINT64_MAX;
            mask[1] = UINT64_MAX;
        } else if (sizeof(Type) == 4) {
            mask[1] = 0;
            mask[0] = UINT64_MAX;
        }
        Copy<Type, false>(outputLocal, inputLocal, mask, repeat, repeatParams);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    pipe_barrier(PIPE_ALL);
    g_sysWorkspaceReserved = GetSysWorkSpacePtr();
}
#define VEC_COPY_TESTCASE(type, testMode)                                                 \
    TEST_F(TestCopy, Copy##type##testMode##Case)                                         \
    {                                                                                      \
        uint32_t dataSize = 512;                                                          \
        uint8_t inputGm[dataSize * sizeof(type)] = {0};                                \
        uint8_t outputGm[dataSize * sizeof(type)] = {0};                               \
        CopyRepeatParams repParams(1, 1, 8, 8);                                            \
        int repeat = dataSize * sizeof(type) / 256;                                       \
                                                             \
        MainVecCopyDemo<type>(outputGm, inputGm, repeat, repParams, testMode); \
        for (uint32_t i = 0; i < dataSize; i++) {                                         \
            EXPECT_EQ(outputGm[i], 0x00);                                                 \
        }                                                                                  \
    }

VEC_COPY_TESTCASE(int32_t, LEVEL0_COUNT_MODE);
VEC_COPY_TESTCASE(uint32_t, LEVEL0_COUNT_MODE);
VEC_COPY_TESTCASE(int16_t, LEVEL0_COUNT_MODE);
VEC_COPY_TESTCASE(uint16_t, LEVEL0_COUNT_MODE);
VEC_COPY_TESTCASE(uint16_t, LEVEL0_BIT_MODE);
VEC_COPY_TESTCASE(uint32_t, LEVEL0_BIT_MODE);
VEC_COPY_TESTCASE(int16_t, LEVEL0_BIT_MODE);
VEC_COPY_TESTCASE(int32_t, LEVEL0_BIT_MODE);