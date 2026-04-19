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

// using namespace std;
using namespace AscendC;

template <typename SrcType> class KernelDuplicate { // 定义kernel类函数
public:
    __aicore__ inline KernelDuplicate() {}
    __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t stackSize, uint32_t dataSize,
        uint64_t maskCounter, uint64_t maskBitHigh, uint64_t maskBitLow, uint8_t repeatTimes,
        uint16_t dstBlockStride, uint8_t dstRepeatStride)
    {
        this->stackSize = stackSize; // 占用空间大小，32Bytre对齐，可能含有脏数据
        this->dataSize = dataSize; // 有效计算数据量
        this->maskCounter = maskCounter;
        this->maskBit[0] = maskBitHigh;
        this->maskBit[1] = maskBitLow;
        this->repeatTimes = repeatTimes;
        this->dstBlockStride = dstBlockStride;
        this->dstRepeatStride = dstRepeatStride;

        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ SrcType*>(srcGm), stackSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ SrcType*>(dstGm), stackSize);

        pipe.InitBuffer(inQueueSrc, 1, stackSize * sizeof(SrcType));
        pipe.InitBuffer(outQueue, 1, stackSize * sizeof(SrcType));
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
        LocalTensor<SrcType> srcLocal = inQueueSrc.AllocTensor<SrcType>();

        DataCopy(srcLocal, srcGlobal, stackSize);

        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<SrcType> dstLocal = outQueue.AllocTensor<SrcType>();
        LocalTensor<SrcType> srcLocal = inQueueSrc.DeQue<SrcType>();

        scalar = srcLocal.GetValue(0);

        event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIDSToV);
        WaitFlag<HardEvent::S_V>(eventIDSToV);
        Duplicate(dstLocal, scalar, maskBit, repeatTimes, dstBlockStride, dstRepeatStride);
        Duplicate(dstLocal, scalar, maskCounter, repeatTimes, dstBlockStride, dstRepeatStride);
        Duplicate(dstLocal, scalar, dataSize);
        outQueue.EnQue<SrcType>(dstLocal);

        inQueueSrc.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<SrcType> dstLocal = outQueue.DeQue<SrcType>();
        DataCopy(dstGlobal, dstLocal, stackSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<SrcType> srcGlobal;
    GlobalTensor<SrcType> dstGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrc;
    TQue<TPosition::VECOUT, 1> outQueue;

    uint64_t maskCounter = 0;
    uint64_t maskBit[2] = {0, 0};

    uint8_t repeatTimes = 0;
    uint16_t dstBlockStride = 1;
    uint8_t dstRepeatStride = 8;

    uint32_t stackSize = 0;
    uint32_t dataSize = 0;
    SrcType scalar = 2;
};

template <typename SrcType>
__aicore__ void DuplicateTest(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t stackSize, uint32_t dataSize,
    uint64_t maskCounter, uint64_t maskBitHigh, uint64_t maskBitLow, uint8_t repeatTimes,
    uint16_t dstBlockStride, uint8_t dstRepeatStride)
{
    KernelDuplicate<SrcType> op;
    op.Init(srcGm, dstGm, stackSize, dataSize, maskCounter, maskBitHigh, maskBitLow, repeatTimes, dstBlockStride, dstRepeatStride);
    op.Process();
}

struct InputParams {
    uint32_t stackSize;
    uint32_t dataSize;
    uint32_t dataTypeSize;
    uint64_t maskCounter;
    uint64_t maskBitHigh;
    uint64_t maskBitLow;
    uint8_t repeatTimes;
    uint16_t dstBlockStride;
    uint8_t dstRepeatStride;
    void (*calFunc)(uint8_t*,  uint8_t*, uint32_t, uint32_t, uint64_t, uint64_t, uint64_t, uint8_t, uint16_t, uint8_t);
};

class DuplicateTestsuite : public ::testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DuplicateTestsuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "DuplicateTestsuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {GlobalMockObject::verify();}
};

#define DUP_PARAMS_NORMAL_TEST(testCaseName, dataType, funcName)                                         \
TEST_F(DuplicateTestsuite, testCaseName)                                                                   \
{                                                                                                            \
    InputParams inputParams{256,  256,  sizeof(dataType), 11, 22, 0, 3, 3, 3};                              \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                                \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                             \
    DuplicateTest<dataType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,                   \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                            \
        inputParams.repeatTimes, inputParams.dstBlockStride, inputParams.dstRepeatStride);                   \
    EXPECT_EQ(outputGm[0], 0x00);                                                                            \
    EXPECT_EQ(outputGm[1], 0x00);                                                                            \
}                                                                                                            \

DUP_PARAMS_NORMAL_TEST(DupTestSuite_check_Func_float, float, DuplicateImpl);
DUP_PARAMS_NORMAL_TEST(DupTestSuite_check_Func_half, half, DuplicateImpl);
DUP_PARAMS_NORMAL_TEST(DupTestSuite_check_Func_int16, int16_t, DuplicateImpl);
DUP_PARAMS_NORMAL_TEST(DupTestSuite_check_Func_int32, int32_t, DuplicateImpl);
DUP_PARAMS_NORMAL_TEST(DupTestSuite_check_Func_uint16, uint16_t, DuplicateImpl);
DUP_PARAMS_NORMAL_TEST(DupTestSuite_check_Func_uint32, uint32_t, DuplicateImpl);


bool DuplicateImplCheckCalCount(const int32_t& calCount)
{
    return (calCount == 256);
}

#define DUP_PARAMS_TEST(testCaseName, dataType, funcName)                                                \
TEST_F(DuplicateTestsuite, testCaseName)                                                                   \
{                                                                                                            \
    InputParams inputParams{256,  256,  sizeof(dataType), 11, 22, 0, 3, 3, 3};                              \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, const dataType&, const int32_t&))                        \
        .times(1)                                                                                            \
        .with(any(), any(), checkWith(DuplicateImplCheckCalCount));                                          \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, const dataType&,                                         \
        uint64_t*, const uint8_t, const uint16_t, const uint8_t))                                            \
        .times(1);                                                                                           \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, const dataType&,                                         \
        uint64_t, const uint8_t, const uint16_t, const uint8_t))                                             \
        .times(1);                                                                                           \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                                \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                             \
    DuplicateTest<dataType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,                   \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                            \
        inputParams.repeatTimes, inputParams.dstBlockStride, inputParams.dstRepeatStride);                   \
    EXPECT_EQ(outputGm[0], 0x00);                                                                            \
    EXPECT_EQ(outputGm[1], 0x00);                                                                            \
}                                                                                                            \

DUP_PARAMS_TEST(DupTestSuite_check_InputParams_float, float, DuplicateImpl);
DUP_PARAMS_TEST(DupTestSuite_check_InputParams_half, half, DuplicateImpl);
DUP_PARAMS_TEST(DupTestSuite_check_InputParams_int16, int16_t, DuplicateImpl);
DUP_PARAMS_TEST(DupTestSuite_check_InputParams_int32, int32_t, DuplicateImpl);
DUP_PARAMS_TEST(DupTestSuite_check_InputParams_uint16, uint16_t, DuplicateImpl);
DUP_PARAMS_TEST(DupTestSuite_check_InputParams_uint32, uint32_t, DuplicateImpl);

template <typename SrcType>
void DataCopyUB2GMImplStub(__gm__ SrcType* dst, __ubuf__ SrcType* src, const DataCopyParams& intriParams)
{
    dst[0] = src[0];
}

#define DUP_CHECK_RESULT_TEST(testCaseName, dataType, funcName)                                         \
TEST_F(DuplicateTestsuite, testCaseName)                                                                  \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(dataType), 11, 22, 0, 3, 3, 3};                           \
    dataType outputStub[inputParams.stackSize] { 0x03 };                                                   \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, const dataType&, const int32_t&))                       \
        .times(1)                                                                                           \
        .with(outBoundP(outputStub));                                                                       \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, const dataType&,                                        \
        uint64_t*, const uint8_t, const uint16_t, const uint8_t))                                           \
        .times(1)                                                                                           \
        .with(outBoundP(outputStub));                                                                       \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, const dataType&,                                        \
        uint64_t, const uint8_t, const uint16_t, const uint8_t))                                            \
        .times(1)                                                                                           \
        .with(outBoundP(outputStub));                                                                       \
    MOCKER(DataCopyUB2GMImpl, void(*)(__gm__ dataType*, __ubuf__ dataType*, const DataCopyParams&))       \
        .times(1)                                                                                           \
        .will(invoke(DataCopyUB2GMImplStub<dataType>));                                                    \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x01 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    DuplicateTest<dataType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,                  \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                           \
        inputParams.repeatTimes, inputParams.dstBlockStride, inputParams.dstRepeatStride);                  \
    dataType *out = reinterpret_cast<dataType*>(outputGm);                                                \
    EXPECT_EQ(out[0], static_cast<dataType>(0x03));                                                                     \
}                                                                                                           \

DUP_CHECK_RESULT_TEST(DupTestSuite_check_Output_float, float, DuplicateImpl);
DUP_CHECK_RESULT_TEST(DupTestSuite_check_Output_half, half, DuplicateImpl);
DUP_CHECK_RESULT_TEST(DupTestSuite_check_Output_int16, int16_t, DuplicateImpl);
DUP_CHECK_RESULT_TEST(DupTestSuite_check_Output_int32, int32_t, DuplicateImpl);
DUP_CHECK_RESULT_TEST(DupTestSuite_check_Output_uint16, uint16_t, DuplicateImpl);
DUP_CHECK_RESULT_TEST(DupTestSuite_check_Output_uint32, uint32_t, DuplicateImpl);
