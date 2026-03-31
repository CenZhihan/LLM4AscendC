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
#include <type_traits>
#include "kernel_operator.h"
// #include "api_check/kernel_cpu_check.h"
#include "mockcpp/mockcpp.hpp"

using namespace std;
using namespace AscendC;

struct TestDataCopyPadParams {
    uint32_t dataSize;
    uint32_t typeSize;
    void (*CalFunc)(uint8_t*, uint8_t*, int32_t);
};

class TestDataCopyPadSuite : public testing::Test, public testing::WithParamInterface<TestDataCopyPadParams> {
protected:
    void SetUp() {}
    void TearDown() {
        AscendC::CheckSyncState();
        GlobalMockObject::verify();
    }
};

template<typename T>
void MainDataCopyPadKernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;

    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    GlobalTensor<half> testGlobal;
    testGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dstGm), dataSize);

    InitOutput(testGlobal, dataSize * sizeof(T) / sizeof(half));
    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();
    DataCopyExtParams copyParams{1, 20 * sizeof(T), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 1, 3};
    DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
    DataCopyPad(dstGlobal, inputLocal, copyParams);
    DataCopyPad(inputLocal, srcGlobal, copyParams, DataCopyPadExtParams<T>());
}

int32_t RaiseStubForDataCopyPad(int32_t input)
{
    return 0;
}

template<typename T>
void MainDataCopyPadKernelWrongPos(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;

    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);
    GlobalTensor<half> testGlobal;
    testGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dstGm), dataSize);

    InitOutput(testGlobal, dataSize * sizeof(T) / sizeof(half));
    TBuf<TPosition::A1> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    DataCopyParams copyParams{1, 20 * sizeof(T), 0, 0};
    DataCopyPadParams padParams{true, 0, 1, 3};
    DataCopyExtParams copyExtParams{1, 20 * sizeof(T), 0, 0, 0};
    DataCopyPadExtParams<T> padExtParams{true, 0, 1, 3};

    MOCKER(raise, int32_t (*)(int32_t)).times(10).will(invoke(RaiseStubForDataCopyPad));
    DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
    DataCopyPad(srcGlobal, inputLocal, copyParams);
    DataCopyPad(inputLocal, srcGlobal, copyExtParams, padExtParams);
    DataCopyPad(dstGlobal, inputLocal, copyExtParams);
    DataCopyPad(inputLocal, srcGlobal, copyExtParams, DataCopyPadExtParams<T>());

    TBuf<TPosition::A2> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<T> inputLocal2 = tbuf2.Get<T>();

    DataCopyPad(inputLocal2, srcGlobal, copyParams, padParams);
    DataCopyPad(srcGlobal, inputLocal2, copyParams);
    DataCopyPad(inputLocal2, srcGlobal, copyExtParams, padExtParams);
    DataCopyPad(dstGlobal, inputLocal2, copyExtParams);
    DataCopyPad(inputLocal2, srcGlobal, copyExtParams, DataCopyPadExtParams<T>());
    SetAtomicNone();
}

template<typename T>
void MainDataCopyPadKernelB64(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;

    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    GlobalTensor<half> testGlobal;
    testGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dstGm), dataSize);

    InitOutput(testGlobal, dataSize * sizeof(T) / sizeof(half));
    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();
    for (int i = 0; i < 2; ++i) {
        if (i == 0) {
            DataCopyParams copyParams{1, 20 * sizeof(T), 0, 0};
            DataCopyPadParams padParams{true, 0, 1, 0};
            DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
            padParams = {true, 2, 1, 0};
            DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
            padParams = {true, 1, 2, 0};
            DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
            padParams = {false, 1, 1, 0};
            DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
            padParams = {true, 2, 2, 0};
            DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
            padParams = {false, 1, 1, 0};
            DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
            padParams = {false, 0, 0, 0};
            DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
            DataCopyPad(dstGlobal, inputLocal, copyParams);
        } else {
            DataCopyExtParams copyParams{1, 20 * sizeof(T), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 1, 0};
            DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
            padParams = {true, 2, 1, 0};
            DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
            padParams = {true, 1, 2, 0};
            DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
            padParams = {false, 1, 1, 0};
            DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
            padParams = {true, 2, 2, 0};
            DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
            padParams = {false, 0, 0, 0};
            DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
            DataCopyPad(inputLocal, srcGlobal, copyParams, DataCopyPadExtParams<T>());
            DataCopyPad(dstGlobal, inputLocal, copyParams);
        }
    }
}
INSTANTIATE_TEST_CASE_P(TEST_DATA_COPY_PAD, TestDataCopyPadSuite,
    ::testing::Values(TestDataCopyPadParams { 64, 4, MainDataCopyPadKernel<int32_t> },
    TestDataCopyPadParams { 64, 2, MainDataCopyPadKernel<int16_t> },
    TestDataCopyPadParams { 64, 4, MainDataCopyPadKernel<float> },
    TestDataCopyPadParams { 64, 2, MainDataCopyPadKernel<half> },
    TestDataCopyPadParams { 64, 1, MainDataCopyPadKernel<int8_t> },
    TestDataCopyPadParams { 64, 1, MainDataCopyPadKernel<uint8_t> },
    TestDataCopyPadParams { 64, 8, MainDataCopyPadKernelB64<int64_t> },
    TestDataCopyPadParams { 64, 8, MainDataCopyPadKernelB64<uint64_t> },
    TestDataCopyPadParams { 64, 2, MainDataCopyPadKernelWrongPos<int16_t> }
    ));

TEST_P(TestDataCopyPadSuite, TestDataCopyPadCases)
{
    AscendC::SetGCoreType(2);
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.typeSize];
    uint8_t dstGm[param.dataSize * param.typeSize];
    param.CalFunc(srcGm, dstGm, param.dataSize);
    for (uint32_t i = 0; i < 20; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
    AscendC::SetGCoreType(0);
}

class DataCopyUB2L0CTestSuite : public ::testing::Test {
protected:
    virtual void SetUp()
    {
        SetGCoreType(2);
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        SetGCoreType(0);
    }
};
 
TEST_F(DataCopyUB2L0CTestSuite, testCaseDataCopyUB2L0C)
{
    LocalTensor<uint16_t> srcGm;
    LocalTensor<uint16_t> dstGm;
    DataCopyParams params;
    DataCopyEnhancedParams enhancedParams;
    MOCKER(raise).stubs().will(returnValue(static_cast<int>(0)));
    DataCopyUB2L0CIntf(dstGm, srcGm, params, enhancedParams);
}

