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

// using namespace std;
using namespace AscendC;

template <typename T>
__global__ __aicore__ void MainDataCopySimple(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm,
    int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::A1> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> srcL1 = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> srcUb = tbuf1.Get<T>();

    DataCopy(srcUb, srcGlobal, dataSize);
    pipe_barrier(PIPE_ALL);
    DataCopy(dstGlobal, srcL1, dataSize);
    pipe_barrier(PIPE_ALL);
    DataCopy(dstGlobal, srcUb, dataSize);
    pipe_barrier(PIPE_ALL);
}

struct DataCopyTestParams {
    int32_t dataSize;
    int32_t dataBitSize;
    void (*cal_func)(uint8_t*, uint8_t*, int32_t);
};

class DataCopySimpleTestsuite : public testing::Test, public testing::WithParamInterface<DataCopyTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_DATA_COPY_SIMPLE, DataCopySimpleTestsuite,
    ::testing::Values(DataCopyTestParams { 512, 4, MainDataCopySimple<float> },
    DataCopyTestParams { 512, 1, MainDataCopySimple<uint8_t> },
    DataCopyTestParams { 512, 2, MainDataCopySimple<uint16_t> },
    DataCopyTestParams { 512, 4, MainDataCopySimple<uint32_t> },
    DataCopyTestParams { 512, 8, MainDataCopySimple<uint64_t> }));

TEST_P(DataCopySimpleTestsuite, DataCopy610LiteSimpleTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.dataBitSize];
    uint8_t dstGm[param.dataSize * param.dataBitSize];

    param.cal_func(srcGm, dstGm, param.dataSize);
    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

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

template<typename T, bool copyByUb>
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

    if constexpr(copyByUb) {
        TBuf<TPosition::VECCALC> tbuf;
        tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
        LocalTensor<T> inputLocal = tbuf.Get<T>();
        DataCopyExtParams copyParams{1, 20 * sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 1, 3};
        DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
        DataCopyPad(dstGlobal, inputLocal, copyParams);
        DataCopyPad(inputLocal, srcGlobal, copyParams, DataCopyPadExtParams<T>());
    } else {
        TBuf<TPosition::A1> tbuf;
        tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
        LocalTensor<T> inputLocal = tbuf.Get<T>();
        DataCopyExtParams copyParams{1, 20 * sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 1, 3};
        DataCopyPad(inputLocal, srcGlobal, copyParams, padParams);
        DataCopyPad(dstGlobal, inputLocal, copyParams);
        DataCopyPad(inputLocal, srcGlobal, copyParams, DataCopyPadExtParams<T>());
    }
}

INSTANTIATE_TEST_CASE_P(TEST_DATA_COPY_PAD, TestDataCopyPadSuite,
    ::testing::Values(
    TestDataCopyPadParams { 64, 4, MainDataCopyPadKernel<int32_t, true> },
    TestDataCopyPadParams { 64, 4, MainDataCopyPadKernel<uint32_t, true> },
    TestDataCopyPadParams { 64, 2, MainDataCopyPadKernel<int16_t, true> },
    TestDataCopyPadParams { 64, 2, MainDataCopyPadKernel<uint16_t, true> },
    TestDataCopyPadParams { 64, 4, MainDataCopyPadKernel<float, true> },
    TestDataCopyPadParams { 64, 2, MainDataCopyPadKernel<half, true> },
    TestDataCopyPadParams { 64, 1, MainDataCopyPadKernel<int8_t, true> },
    TestDataCopyPadParams { 64, 1, MainDataCopyPadKernel<uint8_t, true> },
    TestDataCopyPadParams { 64, 4, MainDataCopyPadKernel<int32_t, false> },
    TestDataCopyPadParams { 64, 4, MainDataCopyPadKernel<uint32_t, false> },
    TestDataCopyPadParams { 64, 2, MainDataCopyPadKernel<int16_t, false> },
    TestDataCopyPadParams { 64, 2, MainDataCopyPadKernel<uint16_t, false> },
    TestDataCopyPadParams { 64, 4, MainDataCopyPadKernel<float, false> },
    TestDataCopyPadParams { 64, 2, MainDataCopyPadKernel<half, false> },
    TestDataCopyPadParams { 64, 1, MainDataCopyPadKernel<int8_t, false> },
    TestDataCopyPadParams { 64, 1, MainDataCopyPadKernel<uint8_t, false> }));

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

template <typename T>
class SetPadValueTest {
public:
    __aicore__ inline SetPadValueTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, uint32_t n1, uint32_t n2)
    {
        mN1 = n1;
        mN2 = n2;
        mN2Align = n2 % 32 == 0 ? n2 : (n2 / 32 + 1) * 32;
        mSrcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        mDstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);

        mPipe.InitBuffer(mQueInSrc, 1, mN1 * mN2Align * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<T> srcLocal = mQueInSrc.AllocTensor<T>();
        DataCopyExtParams dataCopyExtParams;
        DataCopyPadExtParams<T> padParams;

        dataCopyExtParams.blockCount = mN1;
        dataCopyExtParams.blockLen = mN2 * sizeof(T);
        dataCopyExtParams.srcStride = 0;
        dataCopyExtParams.dstStride = 0;

        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 1;

        SetPadValue<T, TPosition::MAX>(static_cast<T>(37));
        pipe_barrier(PIPE_ALL);
        DataCopyPad(srcLocal, mSrcGlobal, dataCopyExtParams, padParams);
        mQueInSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        ;
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = mQueInSrc.DeQue<T>();
        DataCopy(mDstGlobal, dstLocal, mN1 * mN2Align);
        mQueInSrc.FreeTensor(dstLocal);
    }
private:
    TPipe mPipe;
    uint32_t mN1;
    uint32_t mN2;
    uint32_t mN2Align;
    GlobalTensor<T> mSrcGlobal;
    GlobalTensor<T> mDstGlobal;
    TQue<TPosition::VECIN, 1> mQueInSrc;
}; // class SetPadValueTest

template <typename T>
__global__ __aicore__ void testSetPadValue(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t n1, uint32_t n2)
{
    SetPadValueTest<T> op;
    op.Init(dstGm, srcGm, n1, n2);
    op.Process();
}

struct SetPadValueParams {
    uint32_t n1;
    uint32_t n2;
    uint32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t);
};

class SetPadValueTestsuite : public testing::Test,
    public testing::WithParamInterface<SetPadValueParams> {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_SETPADVALUE, SetPadValueTestsuite,
    ::testing::Values(SetPadValueParams { 32, 31, 2, testSetPadValue<half>}));

TEST_P(SetPadValueTestsuite, testSetPadValue)
{
    auto param = GetParam();
    uint32_t n1 = param.n1;
    uint32_t n2 = param.n2;
    uint32_t n2Align = n2 % 32 == 0 ? n2 : (n2 / 32 + 1) * 32;
    uint8_t srcGm[n1 * n2 * param.typeSize] = {0};
    uint8_t dstGm[n1 * n2Align * param.typeSize] = {0};
    param.cal_func(dstGm, srcGm, n1, n2);
    
    for (int32_t i = 0; i < n1 * n2Align; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}