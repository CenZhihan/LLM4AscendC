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

class TEST_SELECT : public testing::Test {
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
void MainVecSelectLevel2Demo(__gm__ uint8_t* __restrict__ srdstGm, __gm__ uint8_t* __restrict__ selMaskGm,
    __gm__ uint8_t* __restrict__ src0Gm, __gm__ uint8_t* __restrict__ src1Gm, SELMODE selMode, uint32_t dataSize,
    uint32_t selMaskSize, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<T> input1Global;
    GlobalTensor<uint8_t> selMaskGlobal;
    GlobalTensor<T> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    input1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1Gm), dataSize);
    selMaskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(selMaskGm), selMaskSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srdstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> input0Local = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> input1Local = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, selMaskSize * sizeof(uint8_t));
    LocalTensor<uint8_t> selMaskLocal = tbuf2.Get<uint8_t>();
    
    TBuf<TPosition::VECCALC> tbuf3;
    tpipe.InitBuffer(tbuf3, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf3.Get<T>();

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);
    DataCopy(selMaskLocal, selMaskGlobal, selMaskSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if (testMode == LEVEL2) {
        Select(outputLocal, selMaskLocal, input0Local, input1Local, selMode, dataSize);
    } else if (testMode == LEVEL0_BIT_MODE) {
        uint8_t repeatTime = dataSize * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
        uint64_t mask[2];
        if (sizeof(T) == 2) {
            mask[0] = UINT64_MAX;
            mask[1] = UINT64_MAX;
        } else if (sizeof(T) == 4) {
            mask[0] = UINT64_MAX;
            mask[1] = 0;
        }
        if (selMode == SELMODE::VSEL_TENSOR_SCALAR_MODE) {
            Select(outputLocal, selMaskLocal, input0Local, static_cast<T>(0), selMode, mask, repeatTime,
                { 1, 1, 1, 8, 8, 8 });
        } else {
            Select(outputLocal, selMaskLocal, input0Local, input1Local, selMode, mask, repeatTime,
                { 1, 1, 1, 8, 8, 8 });
        }
    } else if (testMode == LEVEL0_COUNT_MODE) {
        uint8_t repeatTime = dataSize * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
        uint64_t mask = 0;
        if (sizeof(T) == 2) {
            mask = 128;
        } else if (sizeof(T) == 4) {
            mask = 64;
        }
        if (selMode == SELMODE::VSEL_TENSOR_SCALAR_MODE) {
            Select(outputLocal, selMaskLocal, input0Local, static_cast<T>(0), selMode, mask, repeatTime,
                { 1, 1, 1, 8, 8, 8 });
        } else {
            Select(outputLocal, selMaskLocal, input0Local, input1Local, selMode, mask, repeatTime,
                { 1, 1, 1, 8, 8, 8 });
        }
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    pipe_barrier(PIPE_ALL);
}
#define VEC_SELECT_LEVEL2_TESTCASE(dataType, selMode1, testMode1)                                              \
    TEST_F(TEST_SELECT, Select##dataType##selMode1##testMode1##Case)                                           \
    {                                                                                                           \
        uint32_t dataSize = 256;                                                                               \
        uint32_t selMaskSize = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));                            \
        uint8_t input0Gm[dataSize * sizeof(dataType)];                                                       \
        uint8_t input1Gm[dataSize * sizeof(dataType)];                                                       \
        uint8_t selMaskGm[dataSize];                                                                         \
        uint8_t outputGm[dataSize * sizeof(dataType)];                                                       \
                                                                                                                \
        MainVecSelectLevel2Demo<dataType>(outputGm, selMaskGm, input0Gm, input1Gm, SELMODE::selMode1, \
            dataSize, selMaskSize, testMode1);                                                               \
                                                                                                                \
        for (uint32_t i = 0; i < dataSize; i++) {                                                              \
            EXPECT_EQ(outputGm[i], 0x00);                                                                      \
        }                                                                                                       \
    }
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_CMPMASK_SPR, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_CMPMASK_SPR, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE);

VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE);

VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_SCALAR_MODE, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_SCALAR_MODE, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_TENSOR_MODE, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_TENSOR_MODE, LEVEL2);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE);