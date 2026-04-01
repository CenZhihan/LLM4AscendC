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
// #include "api_check/kernel_cpu_check.h"
#include "mockcpp/mockcpp.hpp"

using namespace std;
using namespace AscendC;

struct TestDataCopyPadParams {
    uint32_t dataSize;
    uint32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, int32_t);
};

class TestDataCopyPadSuite : public testing::Test, public testing::WithParamInterface<TestDataCopyPadParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

template<typename T>
void mainDataCopyPadKernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
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

    TBuf<TPosition::VECCALC> tbuf3;
    tpipe.InitBuffer(tbuf3, dataSize * sizeof(T));
    LocalTensor<T> inputLocal3 = tbuf3.Get<T>();
    DataCopyPad(inputLocal3, srcGlobal, copyExtParams, padExtParams);
    DataCopyPad(dstGlobal, inputLocal3, copyExtParams);
    DataCopyPad(inputLocal3, srcGlobal, copyExtParams, DataCopyPadExtParams<T>());

    SetAtomicNone();
}

INSTANTIATE_TEST_CASE_P(TEST_DATA_COPY_PAD, TestDataCopyPadSuite,
    ::testing::Values(TestDataCopyPadParams { 64, 4, mainDataCopyPadKernel<int32_t> },
    TestDataCopyPadParams { 64, 2, mainDataCopyPadKernel<int16_t> },
    TestDataCopyPadParams { 64, 4, mainDataCopyPadKernel<float> },
    TestDataCopyPadParams { 64, 2, mainDataCopyPadKernel<half> },
    TestDataCopyPadParams { 64, 1, mainDataCopyPadKernel<int8_t> },
    TestDataCopyPadParams { 64, 1, mainDataCopyPadKernel<uint8_t> },
    TestDataCopyPadParams { 64, 2, MainDataCopyPadKernelWrongPos<int16_t> }
    ));

TEST_P(TestDataCopyPadSuite, TestDataCopyPadCases)
{
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.typeSize];
    uint8_t dstGm[param.dataSize * param.typeSize];
    param.cal_func(srcGm, dstGm, param.dataSize);
    for (uint32_t i = 0; i < 20; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

class DataCopyUB2L1TestSuite : public ::testing::Test {
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
 
TEST_F(DataCopyUB2L1TestSuite, testCaseDataCopyUB2L1)
{
    LocalTensor<uint16_t> srcGm;
    LocalTensor<uint16_t> dstGm;
    DataCopyParams params;
    MOCKER(raise).stubs().will(returnValue(static_cast<int>(0)));
    DataCopyL12UBImpl((__cbuf__ uint16_t*)dstGm.GetPhyAddr(), (__ubuf__ uint16_t*)srcGm.GetPhyAddr(), params);
}

class DataCopyL12BTTestSuite : public ::testing::Test {
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
 
TEST_F(DataCopyL12BTTestSuite, testCaseDataCopyL12BT)
{
    LocalTensor<uint16_t> srcGm;
    LocalTensor<uint16_t> dstGm;
    DataCopyParams params;
    MOCKER(raise).stubs().will(returnValue(static_cast<int>(0)));
    DataCopyL12BTImpl((uint64_t)dstGm.GetPhyAddr(), (__cbuf__ uint16_t*)srcGm.GetPhyAddr(), static_cast<uint16_t>(0), params);
}