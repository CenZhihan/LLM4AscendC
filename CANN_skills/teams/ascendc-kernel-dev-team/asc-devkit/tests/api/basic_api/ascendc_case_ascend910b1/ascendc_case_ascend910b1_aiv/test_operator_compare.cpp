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

#include "kernel_log.h"
static uint8_t testRes = 1;      // 全局变量记录运行结果, 如果进入ASCENDC_ASSERT报错，会被置为0
// 重定义ASCENDC_ASSERT，不Abort，仅修改全局变量通知进入报错分支
#undef ASCENDC_ASSERT
#define ASCENDC_ASSERT(cond, behavior) \
    do {                               \
        if (!(cond)) {                 \
            testRes = 0;              \
            behavior;                  \
        }                              \
    } while (0)

#undef ASCENDC_REPORT_CHECK_ERROR
#define ASCENDC_REPORT_CHECK_ERROR(apiMsg, funcType)   \
    do {                                                \
        testRes = 0;                                   \
    } while (0)

#include "kernel_utils.h"
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;

enum TestMode {
    LEVEL2,
    LEVEL0_BIT_MODE,
    LEVEL0_COUNT_MODE,
};

class TestCompare : public testing::Test {
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
void MainVecCompareDemo(__gm__ uint8_t* __restrict__ selMaskGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ src1Gm, CMPMODE cmpMode, uint32_t dataSize, uint32_t selMaskSize,
    TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<T> input1Global;
    GlobalTensor<uint8_t> selMaskGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    input1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1Gm), dataSize);
    selMaskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(selMaskGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> input0Local = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> input1Local = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, selMaskSize * sizeof(uint8_t));
    LocalTensor<uint8_t> selMaskLocal = tbuf2.Get<uint8_t>();

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if (testMode == LEVEL2) {
        Compare(selMaskLocal, input0Local, input1Local, cmpMode, dataSize);
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
        Compare(selMaskLocal, input0Local, input1Local, cmpMode, mask, repeatTime, { 0, 1, 1, 0, 8, 8 });
        Compare(selMaskLocal, input0Local, input1Local, cmpMode, mask, repeatTime, { 0, 2, 2, 0, 16, 16 });
        Compare(input0Local, input1Local, cmpMode, mask, {1, 1, 1, 8, 8, 8});
        Compare(input0Local, input1Local, cmpMode, mask, {1, 2, 2, 8, 16, 16});
    } else if (testMode == LEVEL0_COUNT_MODE) {
        uint8_t repeatTime = dataSize * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
        uint64_t mask = 0;
        if (sizeof(T) == 2) {
            mask = 128;
        } else if (sizeof(T) == 4) {
            mask = 64;
        }
        Compare(selMaskLocal, input0Local, input1Local, cmpMode, mask, repeatTime, { 0, 1, 1, 0, 8, 8 });
        Compare(input0Local, input1Local, cmpMode, mask, {1, 1, 1, 8, 8, 8});
        Compare(input0Local, input1Local, cmpMode, mask, {1, 2, 2, 8, 16, 16});
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(selMaskGlobal, selMaskLocal, selMaskSize);

    pipe_barrier(PIPE_ALL);

    int32_t tmp = g_coreType;
    g_coreType = AscendC::AIC_TYPE;
    LocalTensor<float> src;
    LocalTensor<float> dst;
    SetCmpMask(src);
    GetCmpMask(dst);
    g_coreType = tmp;

    pipe_barrier(PIPE_ALL);
}
#define VEC_CMP_TESTCASE(dataType, relationOp, testMode)                                                \
    TEST_F(TestCompare, Compare##dataType##relationOp##testMode##Case)                                 \
    {                                                                                                      \
        uint32_t dataSize = 256;                                                                          \
        uint32_t selMaskSize = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));                    \
        uint8_t input0Gm[dataSize * sizeof(dataType)];                                                  \
        uint8_t input1Gm[dataSize * sizeof(dataType)];                                                  \
        uint8_t outputGm[dataSize];                                                                      \
                                                                                                           \
        MainVecCompareDemo<dataType>(outputGm, input0Gm, input1Gm, CMPMODE::relationOp, dataSize, \
            selMaskSize, testMode);                                                                     \
                                                                                                           \
        for (uint32_t i = 0; i < selMaskSize; i++) {                                                     \
            EXPECT_EQ(outputGm[i], 0x00);                                                                 \
        }                                                                                                  \
    }
VEC_CMP_TESTCASE(float, LT, LEVEL2);
VEC_CMP_TESTCASE(half, LT, LEVEL2);
VEC_CMP_TESTCASE(float, GT, LEVEL2);
VEC_CMP_TESTCASE(half, GT, LEVEL2);
VEC_CMP_TESTCASE(float, EQ, LEVEL2);
VEC_CMP_TESTCASE(half, EQ, LEVEL2);
VEC_CMP_TESTCASE(float, LE, LEVEL2);
VEC_CMP_TESTCASE(half, LE, LEVEL2);
VEC_CMP_TESTCASE(float, GE, LEVEL2);
VEC_CMP_TESTCASE(half, GE, LEVEL2);
VEC_CMP_TESTCASE(float, NE, LEVEL2);
VEC_CMP_TESTCASE(half, NE, LEVEL2);
VEC_CMP_TESTCASE(float, LT, LEVEL0_BIT_MODE);
VEC_CMP_TESTCASE(half, LT, LEVEL0_BIT_MODE);
VEC_CMP_TESTCASE(float, GT, LEVEL0_BIT_MODE);
VEC_CMP_TESTCASE(half, GT, LEVEL0_BIT_MODE);
VEC_CMP_TESTCASE(float, EQ, LEVEL0_BIT_MODE);
VEC_CMP_TESTCASE(half, EQ, LEVEL0_BIT_MODE);
VEC_CMP_TESTCASE(float, LE, LEVEL0_BIT_MODE);
VEC_CMP_TESTCASE(half, LE, LEVEL0_BIT_MODE);
VEC_CMP_TESTCASE(float, GE, LEVEL0_BIT_MODE);
VEC_CMP_TESTCASE(half, GE, LEVEL0_BIT_MODE);
VEC_CMP_TESTCASE(float, NE, LEVEL0_BIT_MODE);
VEC_CMP_TESTCASE(half, NE, LEVEL0_BIT_MODE);
VEC_CMP_TESTCASE(float, LT, LEVEL0_COUNT_MODE);
VEC_CMP_TESTCASE(half, LT, LEVEL0_COUNT_MODE);
VEC_CMP_TESTCASE(float, GT, LEVEL0_COUNT_MODE);
VEC_CMP_TESTCASE(half, GT, LEVEL0_COUNT_MODE);
VEC_CMP_TESTCASE(float, EQ, LEVEL0_COUNT_MODE);
VEC_CMP_TESTCASE(half, EQ, LEVEL0_COUNT_MODE);
VEC_CMP_TESTCASE(float, LE, LEVEL0_COUNT_MODE);
VEC_CMP_TESTCASE(half, LE, LEVEL0_COUNT_MODE);
VEC_CMP_TESTCASE(float, GE, LEVEL0_COUNT_MODE);
VEC_CMP_TESTCASE(half, GE, LEVEL0_COUNT_MODE);
VEC_CMP_TESTCASE(float, NE, LEVEL0_COUNT_MODE);
VEC_CMP_TESTCASE(half, NE, LEVEL0_COUNT_MODE);
