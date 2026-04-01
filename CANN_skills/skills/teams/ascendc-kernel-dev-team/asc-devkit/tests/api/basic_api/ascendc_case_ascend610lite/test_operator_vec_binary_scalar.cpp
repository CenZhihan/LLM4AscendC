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

enum class OpType{
    ADDS = 0, // Adds, Muls, Maxs, Mins
    MULS = 1,
    MAXS = 2,
    MINS = 3,
    SHIFTLEFT = 4,
    SHIFTRIGHT = 5,
    LEAKY_RELU = 6, // LeakyRelu
};

template <typename SrcType> class KernelBinaryScalar { // 定义kernel类函数
public:
    __aicore__ inline KernelBinaryScalar() {}
    __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t stackSize, uint32_t dataSize,
        uint64_t maskCounter, uint64_t maskBitHigh, uint64_t maskBitLow, uint8_t repeatTimes, UnaryRepeatParams& repeatParams, OpType type)
    {
        this->stackSize = stackSize; // 占用空间大小，32Bytre对齐，可能含有脏数据
        this->dataSize = dataSize; // 有效计算数据量
        this->maskCounter = maskCounter;
        this->maskBit[0] = maskBitHigh;
        this->maskBit[1] = maskBitLow;
        this->repeatTimes = repeatTimes;
        this->repeatParams = repeatParams;
        this->opType = type;

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

        if (opType == OpType::ADDS) {
            Adds(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
            Adds(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            Adds(dstLocal, srcLocal, scalar, dataSize);
        } else if (opType == OpType::MULS) {
            Muls(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
            Muls(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            Muls(dstLocal, srcLocal, scalar, dataSize);
        } else if (opType == OpType::MAXS) {
            Maxs(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
            Maxs(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            Maxs(dstLocal, srcLocal, scalar, dataSize);
        } else if (opType == OpType::MINS) {
            Mins(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
            Mins(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            Mins(dstLocal, srcLocal, scalar, dataSize);
        } else if (opType == OpType::SHIFTLEFT) {
            ShiftLeft(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
            ShiftLeft(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            ShiftLeft(dstLocal, srcLocal, scalar, dataSize);
        } else if (opType == OpType::SHIFTRIGHT) {
            ShiftRight(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams, false);
            ShiftRight(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams, false);
            ShiftRight(dstLocal, srcLocal, scalar, dataSize);
        } else {
            LeakyRelu(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
            LeakyRelu(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            LeakyRelu(dstLocal, srcLocal, scalar, dataSize);            
        }
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
    UnaryRepeatParams repeatParams { 1, 1, 8, 8 };

    uint32_t stackSize = 0;
    uint32_t dataSize = 0;
    SrcType scalar = 2;
    OpType opType;
};

template <typename SrcType>
__aicore__ void BinaryScalarTest(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t stackSize, uint32_t dataSize,
    uint64_t maskCounter, uint64_t maskBitHigh, uint64_t maskBitLow, uint8_t repeatTimes, UnaryRepeatParams& repeatParams, OpType opType)
{
    KernelBinaryScalar<SrcType> op;
    op.Init(srcGm, dstGm, stackSize, dataSize, maskCounter, maskBitHigh, maskBitLow, repeatTimes, repeatParams, opType);
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
    UnaryRepeatParams repeatParams;
    OpType opType;
};

class BinaryScalarTestsuite : public ::testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BinaryScalarTestsuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "BinaryScalarTestsuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {GlobalMockObject::verify();}
};

#define SCALAR_BINARY_FUNC_TEST(testCaseName, dataType, funcName, opType1)                              \
TEST_F(BinaryScalarTestsuite, testCaseName)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(dataType), 11, 22, 0, 3, { 3, 1, 3, 1}, (OpType)opType1}; \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    BinaryScalarTest<dataType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,               \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                           \
        inputParams.repeatTimes, inputParams.repeatParams, inputParams.opType);                             \
    EXPECT_EQ(outputGm[0], 0x00);                                                                           \
    EXPECT_EQ(outputGm[1], 0x00);                                                                           \
}                                                                                                           \

#define SCALAR_BINARY_SHIFTRIGHT_FUNC_TEST(testCaseName, dataType, funcName, opType1)                   \
TEST_F(BinaryScalarTestsuite, testCaseName)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(dataType), 11, 22, 0, 3, { 3, 1, 3, 1}, (OpType)opType1}; \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    BinaryScalarTest<dataType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,               \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                           \
        inputParams.repeatTimes, inputParams.repeatParams, inputParams.opType);                             \
    EXPECT_EQ(outputGm[0], 0x00);                                                                           \
}

SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_float, float, AddsImpl, 0);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_half, half, AddsImpl, 0);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_int16, int16_t, AddsImpl, 0);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_int32, int32_t, AddsImpl, 0);

SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_float, float, MulsImpl, 1);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_half, half, MulsImpl, 1);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_int16, int16_t, MulsImpl, 1);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_int32, int32_t, MulsImpl, 1);

SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_float, float, MaxsImpl, 2);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_half, half, MaxsImpl, 2);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_int16, int16_t, MaxsImpl, 2);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_int32, int32_t, MaxsImpl, 2);

SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_float, float, MinsImpl, 3);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_half, half, MinsImpl, 3);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_int16, int16_t, MinsImpl, 3);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_int32, int32_t, MinsImpl, 3);

SCALAR_BINARY_FUNC_TEST(ShiftLeftTestSuite_check_Func_uint16, uint16_t, ShiftLeftImpl, 4);
SCALAR_BINARY_FUNC_TEST(ShiftLeftTestSuite_check_Func_uint32, uint32_t, ShiftLeftImpl, 4);
SCALAR_BINARY_FUNC_TEST(ShiftLeftTestSuite_check_Func_int16, int16_t, ShiftLeftImpl, 4);
SCALAR_BINARY_FUNC_TEST(ShiftLeftTestSuite_check_Func_int32, int32_t, ShiftLeftImpl, 4);

SCALAR_BINARY_FUNC_TEST(LeakyReluTestSuite_check_Func_float, float, LeakyReluImpl, 6);
SCALAR_BINARY_FUNC_TEST(LeakyReluTestSuite_check_Func_half, half, LeakyReluImpl, 6);

SCALAR_BINARY_SHIFTRIGHT_FUNC_TEST(ShiftRightTestSuite_check_Func_uint16, uint16_t, ShiftRightImpl, 5);
SCALAR_BINARY_SHIFTRIGHT_FUNC_TEST(ShiftRightTestSuite_check_Func_uint32, uint32_t, ShiftRightImpl, 5);
SCALAR_BINARY_SHIFTRIGHT_FUNC_TEST(ShiftRightTestSuite_check_Func_int16, int16_t, ShiftRightImpl, 5);
SCALAR_BINARY_SHIFTRIGHT_FUNC_TEST(ShiftRightTestSuite_check_Func_int32, int32_t, ShiftRightImpl, 5);

template <typename SrcType>
bool BinaryScalarCheckSrcValue(__ubuf__ SrcType* src)
{
    return (src[0] == static_cast<SrcType>(0));
}

bool BinaryScalarCheckCalCount(const int32_t& calCount)
{
    return (calCount == 1020);
}

#define SCALAR_BINARY_TEST(testCaseName, dataType, funcName, opType1)                                   \
TEST_F(BinaryScalarTestsuite, testCaseName)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(dataType), 11, 22, 0, 3, { 3, 1, 3, 1}, (OpType)opType1}; \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType, const int32_t&))         \
        .times(1)                                                                                           \
        .with(any(), checkWith(BinaryScalarCheckSrcValue<dataType>));                                      \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType,                          \
        const uint64_t*, const uint8_t, const UnaryRepeatParams&))                                          \
        .times(1);                                                                                          \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType,                          \
        const uint64_t, const uint8_t, const UnaryRepeatParams&))                                           \
        .times(1);                                                                                          \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    BinaryScalarTest<dataType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,               \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                           \
        inputParams.repeatTimes, inputParams.repeatParams, inputParams.opType);                             \
    EXPECT_EQ(outputGm[0], 0x00);                                                                           \
    EXPECT_EQ(outputGm[1], 0x00);                                                                           \
}                                                                                                           \

#define SCALAR_BINARY_SHIFTRIGHT_TEST(testCaseName, dataType, funcName, opType1)                        \
TEST_F(BinaryScalarTestsuite, testCaseName)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(dataType), 11, 22, 0, 3, { 3, 1, 3, 1}, (OpType)opType1}; \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType, const int32_t&))         \
        .times(1);                                                                                          \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType,                          \
        const uint64_t*, const uint8_t, const UnaryRepeatParams&, bool))                                    \
        .times(1);                                                                                          \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType,                          \
        const uint64_t, const uint8_t, const UnaryRepeatParams&, bool))                                     \
        .times(1);                                                                                          \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    BinaryScalarTest<dataType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,               \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                           \
        inputParams.repeatTimes, inputParams.repeatParams, inputParams.opType);                             \
    EXPECT_EQ(outputGm[0], 0x00);                                                                           \
}

SCALAR_BINARY_TEST(AddsTestSuite_check_InputParams_float, float, AddsImpl, 0);
SCALAR_BINARY_TEST(AddsTestSuite_check_InputParams_half, half, AddsImpl, 0);
SCALAR_BINARY_TEST(AddsTestSuite_check_InputParams_int16, int16_t, AddsImpl, 0);
SCALAR_BINARY_TEST(AddsTestSuite_check_InputParams_int32, int32_t, AddsImpl, 0);

SCALAR_BINARY_TEST(MulsTestSuite_check_InputParams_float, float, MulsImpl, 1);
SCALAR_BINARY_TEST(MulsTestSuite_check_InputParams_half, half, MulsImpl, 1);
SCALAR_BINARY_TEST(MulsTestSuite_check_InputParams_int16, int16_t, MulsImpl, 1);
SCALAR_BINARY_TEST(MulsTestSuite_check_InputParams_int32, int32_t, MulsImpl, 1);

SCALAR_BINARY_TEST(MaxsTestSuite_check_InputParams_float, float, MaxsImpl, 2);
SCALAR_BINARY_TEST(MaxsTestSuite_check_InputParams_half, half, MaxsImpl, 2);
SCALAR_BINARY_TEST(MaxsTestSuite_check_InputParams_int16, int16_t, MaxsImpl, 2);
SCALAR_BINARY_TEST(MaxsTestSuite_check_InputParams_int32, int32_t, MaxsImpl, 2);

SCALAR_BINARY_TEST(MinsTestSuite_check_InputParams_float, float, MinsImpl, 3);
SCALAR_BINARY_TEST(MinsTestSuite_check_InputParams_half, half, MinsImpl, 3);
SCALAR_BINARY_TEST(MinsTestSuite_check_InputParams_int16, int16_t, MinsImpl, 3);
SCALAR_BINARY_TEST(MinsTestSuite_check_InputParams_int32, int32_t, MinsImpl, 3);

SCALAR_BINARY_TEST(ShiftLeftTestSuite_check_InputParams_uint16, uint16_t, ShiftLeftImpl, 4);
SCALAR_BINARY_TEST(ShiftLeftTestSuite_check_InputParams_uint32, uint32_t, ShiftLeftImpl, 4);
SCALAR_BINARY_TEST(ShiftLeftTestSuite_check_InputParams_int16, int16_t, ShiftLeftImpl, 4);
SCALAR_BINARY_TEST(ShiftLeftTestSuite_check_InputParams_int32, int32_t, ShiftLeftImpl, 4);

SCALAR_BINARY_TEST(LeakyReluTestSuite_check_InputParams_float, float, LeakyReluImpl, 6);
SCALAR_BINARY_TEST(LeakyReluTestSuite_check_InputParams_half, half, LeakyReluImpl, 6);

SCALAR_BINARY_SHIFTRIGHT_TEST(ShiftRightTestSuite_check_InputParams_uint16, uint16_t, ShiftRightImpl, 5);
SCALAR_BINARY_SHIFTRIGHT_TEST(ShiftRightTestSuite_check_InputParams_uint32, uint32_t, ShiftRightImpl, 5);
SCALAR_BINARY_SHIFTRIGHT_TEST(ShiftRightTestSuite_check_InputParams_int16, int16_t, ShiftRightImpl, 5);
SCALAR_BINARY_SHIFTRIGHT_TEST(ShiftRightTestSuite_check_InputParams_int32, int32_t, ShiftRightImpl, 5);

template <typename SrcType>
void DataCopyUB2GMImplStub(__gm__ SrcType* dst, __ubuf__ SrcType* src, const DataCopyParams& intriParams)
{
    dst[0] = src[0];
}

template <typename SrcType>
void CopyCbufToGmAlignStub(__gm__ SrcType* dst, __ubuf__ SrcType* src, uint8_t sid, uint16_t nBurst,
    uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap)
{
    dst[0] = src[0];
}

#define SCALAR_BINARY_TEST_VALUE(testCaseName, dataType, funcName, opType1)                             \
TEST_F(BinaryScalarTestsuite, testCaseName)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(dataType), 11, 22, 0, 3, { 3, 1, 3, 1}, (OpType)opType1}; \
    dataType outputStub[inputParams.stackSize] { 0x03 };                                                   \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType, const int32_t&))         \
        .times(1)                                                                                           \
        .with(outBoundP(outputStub));                                                                       \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType,                          \
        const uint64_t*, const uint8_t, const UnaryRepeatParams&))                                          \
        .times(1)                                                                                           \
        .with(outBoundP(outputStub));                                                                       \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType,                          \
        const uint64_t, const uint8_t, const UnaryRepeatParams&))                                           \
        .times(1)                                                                                           \
        .with(outBoundP(outputStub));                                                                       \
    MOCKER(DataCopyUB2GMImpl, void(*)(__gm__ dataType*, __ubuf__ dataType*, const DataCopyParams&))       \
        .times(1)                                                                                           \
        .will(invoke(DataCopyUB2GMImplStub<dataType>));                                                    \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x01 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    BinaryScalarTest<dataType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,               \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                           \
        inputParams.repeatTimes, inputParams.repeatParams, inputParams.opType);                             \
    dataType *out = reinterpret_cast<dataType*>(outputGm);                                                \
    EXPECT_EQ(out[0], static_cast<dataType>(0x03));                                                                     \
}                                                                                                           \

#define SCALAR_BINARY_SHIFTRIGHT_TEST_VALUE(testCaseName, dataType, funcName, opType1)                  \
TEST_F(BinaryScalarTestsuite, testCaseName)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(dataType), 11, 22, 0, 3, { 3, 1, 3, 1}, (OpType)opType1}; \
    dataType outputStub[inputParams.stackSize] { 0x03 };                                                   \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType, const int32_t&))         \
        .times(1)                                                                                           \
        .with(outBoundP(outputStub));                                                                       \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType,                          \
        const uint64_t*, const uint8_t, const UnaryRepeatParams&, bool))                                    \
        .times(1)                                                                                           \
        .with(outBoundP(outputStub));                                                                       \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType,                          \
        const uint64_t, const uint8_t, const UnaryRepeatParams&, bool))                                     \
        .times(1)                                                                                           \
        .with(outBoundP(outputStub));                                                                       \
    MOCKER(DataCopyUB2GMImpl, void(*)(__gm__ dataType*, __ubuf__ dataType*, const DataCopyParams&))       \
        .times(1)                                                                                           \
        .will(invoke(DataCopyUB2GMImplStub<dataType>));                                                    \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x01 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    BinaryScalarTest<dataType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,               \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                           \
        inputParams.repeatTimes, inputParams.repeatParams, inputParams.opType);                             \
    EXPECT_EQ(outputGm[0], 0x03);                                                                           \
}                                                                                                           \

SCALAR_BINARY_TEST_VALUE(AddsTestSuite_check_Output_float, float, AddsImpl, 0);
SCALAR_BINARY_TEST_VALUE(AddsTestSuite_check_Output_half, half, AddsImpl, 0);
SCALAR_BINARY_TEST_VALUE(AddsTestSuite_check_Output_int16, int16_t, AddsImpl, 0);
SCALAR_BINARY_TEST_VALUE(AddsTestSuite_check_Output_int32, int32_t, AddsImpl, 0);

SCALAR_BINARY_TEST_VALUE(MulsTestSuite_check_Output_float, float, MulsImpl, 1);
SCALAR_BINARY_TEST_VALUE(MulsTestSuite_check_Output_half, half, MulsImpl, 1);
SCALAR_BINARY_TEST_VALUE(MulsTestSuite_check_Output_int16, int16_t, MulsImpl, 1);
SCALAR_BINARY_TEST_VALUE(MulsTestSuite_check_Output_int32, int32_t, MulsImpl, 1);

SCALAR_BINARY_TEST_VALUE(MaxsTestSuite_check_Output_float, float, MaxsImpl, 2);
SCALAR_BINARY_TEST_VALUE(MaxsTestSuite_check_Output_half, half, MaxsImpl, 2);
SCALAR_BINARY_TEST_VALUE(MaxsTestSuite_check_Output_int16, int16_t, MaxsImpl, 2);
SCALAR_BINARY_TEST_VALUE(MaxsTestSuite_check_Output_int32, int32_t, MaxsImpl, 2);

SCALAR_BINARY_TEST_VALUE(MinsTestSuite_check_Output_float, float, MinsImpl, 3);
SCALAR_BINARY_TEST_VALUE(MinsTestSuite_check_Output_half, half, MinsImpl, 3);
SCALAR_BINARY_TEST_VALUE(MinsTestSuite_check_Output_int16, int16_t, MinsImpl, 3);
SCALAR_BINARY_TEST_VALUE(MinsTestSuite_check_Output_int32, int32_t, MinsImpl, 3);

SCALAR_BINARY_TEST_VALUE(ShiftLeftTestSuite_check_Output_uint16, uint16_t, ShiftLeftImpl, 4);
SCALAR_BINARY_TEST_VALUE(ShiftLeftTestSuite_check_Output_uint32, uint32_t, ShiftLeftImpl, 4);
SCALAR_BINARY_TEST_VALUE(ShiftLeftTestSuite_check_Output_int16, int16_t, ShiftLeftImpl, 4);
SCALAR_BINARY_TEST_VALUE(ShiftLeftTestSuite_check_Output_int32, int32_t, ShiftLeftImpl, 4);

SCALAR_BINARY_TEST_VALUE(LeakyReluTestSuite_check_Output_float, float, LeakyReluImpl, 6);
SCALAR_BINARY_TEST_VALUE(LeakyReluTestSuite_check_Output_half, half, LeakyReluImpl, 6);


SCALAR_BINARY_SHIFTRIGHT_TEST_VALUE(ShiftRightTestSuite_check_Output_uint16, uint16_t, ShiftRightImpl, 5);
SCALAR_BINARY_SHIFTRIGHT_TEST_VALUE(ShiftRightTestSuite_check_Output_uint32, uint32_t, ShiftRightImpl, 5);
SCALAR_BINARY_SHIFTRIGHT_TEST_VALUE(ShiftRightTestSuite_check_Output_int16, int16_t, ShiftRightImpl, 5);
SCALAR_BINARY_SHIFTRIGHT_TEST_VALUE(ShiftRightTestSuite_check_Output_int32, int32_t, ShiftRightImpl, 5);

#define SCALAR_BINARY_TEST_UNSUPPORT(testCaseName, dataType, opType1)                                    \
TEST_F(BinaryScalarTestsuite, testCaseName)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(dataType), 11, 22, 0, 3, { 3, 1, 3, 1}, (OpType)opType1}; \
    MOCKER(raise, int(*)(int)).times(3).will(returnValue(0));                                               \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    BinaryScalarTest<dataType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,               \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                           \
        inputParams.repeatTimes, inputParams.repeatParams, inputParams.opType);                             \
    EXPECT_EQ(outputGm[0], 0x00);                                                                           \
    EXPECT_EQ(outputGm[1], 0x00);                                                                           \
}                                                                                                           \

#define SCALAR_BINARY_UNSPPORT_TYPE(funcName, opType1, dataType1, dataType2, dataType3, dataType4) \
    SCALAR_BINARY_TEST_UNSUPPORT(funcName##TestSuite_check_DType_##dataType1, dataType1, opType1);   \
    SCALAR_BINARY_TEST_UNSUPPORT(funcName##TestSuite_check_DType_##dataType2, dataType2, opType1);   \
    SCALAR_BINARY_TEST_UNSUPPORT(funcName##TestSuite_check_DType_##dataType3, dataType3, opType1);   \
    SCALAR_BINARY_TEST_UNSUPPORT(funcName##TestSuite_check_DType_##dataType4, dataType4, opType1);   \

SCALAR_BINARY_UNSPPORT_TYPE(Adds, 0, uint32_t, uint16_t, uint8_t, int8_t);
SCALAR_BINARY_UNSPPORT_TYPE(Muls, 1, uint32_t, uint16_t, uint8_t, int8_t);
SCALAR_BINARY_UNSPPORT_TYPE(Maxs, 2, uint32_t, uint16_t, uint8_t, int8_t);
SCALAR_BINARY_UNSPPORT_TYPE(Mins, 3, uint32_t, uint16_t, uint8_t, int8_t);
SCALAR_BINARY_UNSPPORT_TYPE(ShiftLeft, 4, float, half, uint8_t, int8_t);
SCALAR_BINARY_UNSPPORT_TYPE(ShiftRight, 5, float, half, uint8_t, int8_t);