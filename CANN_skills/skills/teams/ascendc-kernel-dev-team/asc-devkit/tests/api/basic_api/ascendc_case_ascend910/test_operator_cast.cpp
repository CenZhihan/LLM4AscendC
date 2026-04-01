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
#include "mockcpp/mockcpp.hpp"

using namespace std;
using namespace AscendC;

enum TestMode {
    LEVEL2,
    LEVEL0_BIT_MODE,
    LEVEL0_COUNT_MODE,
};
enum TestDataCopyCheckMode {
    L12GM,
    L12L0C,
};
class TEST_CAST : public testing::Test {
protected:
    void SetUp() {
        g_coreType = AscendC::AIV_TYPE;
    }
    void TearDown() {
        g_coreType = AscendC::MIX_TYPE;
    }
};

class TEST_CAST_COMPILE : public testing::Test {
protected:
    void SetUp() {
        g_coreType = AscendC::AIV_TYPE;
    }
    void TearDown() {
        g_coreType = AscendC::MIX_TYPE;
        GlobalMockObject::verify();
    }
};

class TEST_DATA_COPY_CHECK : public testing::Test {
protected:
    void SetUp()
    {
        g_coreType = AscendC::AIV_TYPE;
    }
    void TearDown()
    {
        g_coreType = AscendC::MIX_TYPE;
        GlobalMockObject::verify();
    }
};

template <typename DstType, typename SrcType>
void MainVecCastDemo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm, RoundMode roundMode,
    uint32_t dstDataSize, uint32_t srcDataSize, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<SrcType> inputGlobal;
    GlobalTensor<DstType> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ SrcType*>(srcGm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstType*>(dstGm), dstDataSize);

    LocalTensor<SrcType> inputLocal;
    LocalTensor<DstType> outputLocal;

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcDataSize * sizeof(SrcType));
    inputLocal = tbuf.Get<SrcType>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dstDataSize * sizeof(DstType));
    outputLocal = tbuf1.Get<DstType>();

    DataCopy(inputLocal, inputGlobal, srcDataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Cast(outputLocal, inputLocal, roundMode, dstDataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    pipe_barrier(PIPE_ALL);
}

template <typename DstType, typename SrcType>
void MainVecCastDemoDeath(RoundMode roundMode, uint32_t dstDataSize, uint32_t srcDataSize, TestMode testMode)
{
    TPipe tpipe;
    LocalTensor<SrcType> inputLocal;
    LocalTensor<DstType> outputLocal;

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcDataSize * sizeof(SrcType));
    inputLocal = tbuf.Get<SrcType>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dstDataSize * sizeof(DstType));
    outputLocal = tbuf1.Get<DstType>();
    Cast(outputLocal, inputLocal, roundMode, dstDataSize);
}
int32_t RaiseStub(int32_t input)
{
    return 0;
}
#define VEC_CAST_TESTCASE(srcType, dstType, roundMode, testMode)                                      \
    TEST_F(TEST_CAST, Cast##srcType##2##dstType##roundMode##testMode##Case)                           \
    {                                                                                                     \
        uint32_t srcDataSize = 512;                                                                     \
        uint32_t dstDataSize = 512;                                                                     \
        uint8_t inputGm[srcDataSize * sizeof(srcType)];                                               \
        uint8_t outputGm[dstDataSize * sizeof(dstType)];                                              \
                                                                                                          \
        MainVecCastDemo<dstType, srcType>(outputGm, inputGm, RoundMode::roundMode, dstDataSize, \
            srcDataSize, testMode);                                                                    \
                                                                                                          \
        for (uint32_t i = 0; i < dstDataSize; i++) {                                                    \
            EXPECT_EQ(outputGm[i], 0x00);                                                                \
        }                                                                                                 \
    }

#define VEC_CAST_FAILED_TESTCASE(srcType, dstType, roundMode, testMode)                                           \
    TEST_F(TEST_CAST_COMPILE, Cast##srcType##2##dstType##roundMode##testMode##FailedCase)                         \
    {                                                                                                                 \
        uint32_t srcDataSize = 512;                                                                                 \
        uint32_t dstDataSize = 512;                                                                                 \
        MOCKER(raise, int32_t (*)(int32_t)).times(1).will(invoke(RaiseStub));                                         \
        MainVecCastDemoDeath<dstType, srcType>(RoundMode::roundMode, dstDataSize, srcDataSize, testMode); \
    }
VEC_CAST_FAILED_TESTCASE(int8_t, half, CAST_RINT, LEVEL2);
VEC_CAST_FAILED_TESTCASE(uint8_t, half, CAST_RINT, LEVEL2);
VEC_CAST_FAILED_TESTCASE(int32_t, float, CAST_RINT, LEVEL2);
VEC_CAST_FAILED_TESTCASE(half, float, CAST_RINT, LEVEL2);
VEC_CAST_FAILED_TESTCASE(half, int32_t, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(half, int32_t, CAST_NONE, LEVEL2);
VEC_CAST_FAILED_TESTCASE(half, uint8_t, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, half, CAST_RINT, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, half, CAST_FLOOR, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, half, CAST_CEIL, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, half, CAST_ROUND, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, half, CAST_TRUNC, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, int32_t, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(float, int32_t, CAST_NONE, LEVEL2);

VEC_CAST_FAILED_TESTCASE(half, int8_t, CAST_ODD, LEVEL2);
VEC_CAST_FAILED_TESTCASE(half, int8_t, CAST_RINT, LEVEL2);
VEC_CAST_FAILED_TESTCASE(half, uint8_t, CAST_RINT, LEVEL2);

VEC_CAST_TESTCASE(half, int32_t, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(half, int32_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(half, int32_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(half, int32_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(half, int32_t, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(float, int32_t, CAST_RINT, LEVEL2);
VEC_CAST_TESTCASE(float, int32_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(float, int32_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(float, int32_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(float, int32_t, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(int32_t, float, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(half, float, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(float, half, CAST_NONE, LEVEL2);
VEC_CAST_TESTCASE(float, half, CAST_ODD, LEVEL2);

VEC_CAST_TESTCASE(half, int8_t, CAST_NONE, LEVEL2);
VEC_CAST_TESTCASE(half, int8_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(half, int8_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(half, int8_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(half, int8_t, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(half, uint8_t, CAST_NONE, LEVEL2);
VEC_CAST_TESTCASE(half, uint8_t, CAST_FLOOR, LEVEL2);
VEC_CAST_TESTCASE(half, uint8_t, CAST_CEIL, LEVEL2);
VEC_CAST_TESTCASE(half, uint8_t, CAST_ROUND, LEVEL2);
VEC_CAST_TESTCASE(half, uint8_t, CAST_TRUNC, LEVEL2);

VEC_CAST_TESTCASE(int32_t, half, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(uint8_t, half, CAST_NONE, LEVEL2);

VEC_CAST_TESTCASE(int8_t, half, CAST_NONE, LEVEL2);

template <typename DstType, typename SrcType>
void MainVecDataCheckDemo(TestDataCopyCheckMode testMode)
{
    __gm__ DstType *dst;
    __cbuf__ SrcType *src;
    DataCopyParams intriParams;
    DataCopyEnhancedParams enhancedParams;
    if (testMode == TestDataCopyCheckMode::L12GM) {
        DataCopyL12GMImpl<DstType>(dst, src, intriParams);
    } else if (testMode == TestDataCopyCheckMode::L12L0C) {
        DataCopyL12L0CImpl<DstType>(dst, src, intriParams, enhancedParams);
    }
}
#define TEST_DATA_COPY_CHECK_FAILED_TESTCASE(srcType, dstType, testMode)                   \
    TEST_F(TEST_DATA_COPY_CHECK, DATACOPYCHEK##srcType##2##dstType##testMode##FailedCase)  \
    {                                                                                    \
        MOCKER(raise, int32_t (*)(int32_t)).times(1).will(invoke(RaiseStub)); \
        MainVecDataCheckDemo<dstType, srcType>(testMode);                         \
    }
TEST_DATA_COPY_CHECK_FAILED_TESTCASE(int8_t, int8_t, L12GM);
TEST_DATA_COPY_CHECK_FAILED_TESTCASE(int8_t, int8_t, L12L0C);
