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

enum class OpType{
    ADDS = 0, // Adds, Muls, Maxs, Mins
    MULS = 1,
    MAXS = 2,
    MINS = 3,
    SHIFTLEFT = 4,
    SHIFTRIGHT = 5,
    LEAKY_RELU = 6, // LeakyRelu
};

template <typename SrcType, OpType OpType> class KernelBinaryScalar { // 定义kernel类函数
public:
    __aicore__ inline KernelBinaryScalar() {}
    __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t stackSize, uint32_t dataSize,
        uint64_t maskCounter, uint64_t maskBitHigh, uint64_t maskBitLow, uint8_t repeatTimes, UnaryRepeatParams& repeatParams)
    {
        this->stackSize = stackSize; // 占用空间大小，32Bytre对齐，可能含有脏数据
        this->dataSize = dataSize; // 有效计算数据量
        this->maskCounter = maskCounter;
        this->maskBit[0] = maskBitHigh;
        this->maskBit[1] = maskBitLow;
        this->repeatTimes = repeatTimes;
        this->repeatParams = repeatParams;

        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<SrcType>*>(srcGm), stackSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<SrcType>*>(dstGm), stackSize);

        pipe.InitBuffer(inQueueSrc, 1, stackSize * sizeof(PrimT<SrcType>));
        pipe.InitBuffer(outQueue, 1, stackSize * sizeof(PrimT<SrcType>));
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

        if constexpr (OpType == OpType::ADDS) {
            if constexpr (sizeof(PrimT<SrcType>) != 8) {
                Adds(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
                Adds(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
                constexpr static BinaryConfig config = {1};
                Adds<SrcType, false, config>(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            }
            Adds<SrcType, true>(dstLocal, srcLocal, scalar, dataSize);
        } else if constexpr (OpType == OpType::MULS) {
            if constexpr (sizeof(PrimT<SrcType>) != 8) {
                Muls(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
                Muls(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            }
            Muls(dstLocal, srcLocal, scalar, dataSize);
        } else if constexpr (OpType == OpType::MAXS) {
            if constexpr (sizeof(PrimT<SrcType>) != 8) {
                Maxs(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
                Maxs(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            }
            Maxs(dstLocal, srcLocal, scalar, dataSize);
        } else if constexpr (OpType == OpType::MINS) {
            if constexpr (sizeof(PrimT<SrcType>) != 8) {
                Mins(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
                Mins(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            }
            Mins(dstLocal, srcLocal, scalar, dataSize);
        } else if constexpr (OpType == OpType::SHIFTLEFT) {
            ShiftLeft(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
            ShiftLeft(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            ShiftLeft(dstLocal, srcLocal, scalar, dataSize);
        } else if constexpr (OpType == OpType::SHIFTRIGHT) {
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
    PrimT<SrcType> scalar = 2;
};

template <typename SrcType, OpType OpType>
__aicore__ void BinaryScalarTest(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t stackSize, uint32_t dataSize,
    uint64_t maskCounter, uint64_t maskBitHigh, uint64_t maskBitLow, uint8_t repeatTimes, UnaryRepeatParams& repeatParams)
{
    KernelBinaryScalar<SrcType, OpType> op;
    op.Init(srcGm, dstGm, stackSize, dataSize, maskCounter, maskBitHigh, maskBitLow, repeatTimes, repeatParams);
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

#define SCALAR_BINARY_FUNC_TEST(testCaseName, dataType, funcName, opType)                              \
TEST_F(BinaryScalarTestsuite, testCaseName)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(PrimT<dataType>), 11, 22, 0, 3, { 3, 1, 3, 1}};                  \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    BinaryScalarTest<dataType, opType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,      \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                           \
        inputParams.repeatTimes, inputParams.repeatParams);                                                 \
    EXPECT_EQ(outputGm[0], 0x00);                                                                           \
    EXPECT_EQ(outputGm[1], 0x00);                                                                           \
}                                                                                                           \

SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_float, float, AddsImpl, OpType::ADDS);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_half, half, AddsImpl, OpType::ADDS);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_int16, int16_t, AddsImpl, OpType::ADDS);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_int32, int32_t, AddsImpl, OpType::ADDS);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_bfloat16_t, bfloat16_t, AddsImpl, OpType::ADDS);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_int64_t, int64_t, AddsImpl, OpType::ADDS);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_uint64_t, uint64_t, AddsImpl, OpType::ADDS);

SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_float, float, MulsImpl, OpType::MULS);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_half, half, MulsImpl, OpType::MULS);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_int16, int16_t, MulsImpl, OpType::MULS);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_int32, int32_t, MulsImpl, OpType::MULS);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_bfloat16_t, bfloat16_t, MulsImpl, OpType::MULS);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_int64_t, int64_t, MulsImpl, OpType::MULS);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_uint64_t, uint64_t, MulsImpl, OpType::MULS);

SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_float, float, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_half, half, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_int16, int16_t, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_int32, int32_t, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_bfloat16_t, bfloat16_t, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_int64_t, int64_t, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_uint64_t, uint64_t, MaxsImpl, OpType::MAXS);

SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_float, float, MinsImpl, OpType::MINS);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_half, half, MinsImpl, OpType::MINS);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_int16, int16_t, MinsImpl, OpType::MINS);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_int32, int32_t, MinsImpl, OpType::MINS);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_bfloat16_t, bfloat16_t, MinsImpl, OpType::MINS);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_int64_t, int64_t, MinsImpl, OpType::MINS);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_uint64_t, uint64_t, MinsImpl, OpType::MINS);

SCALAR_BINARY_FUNC_TEST(ShiftLeftTestSuite_check_Func_uint16, uint16_t, ShiftLeftImpl, OpType::SHIFTLEFT);
SCALAR_BINARY_FUNC_TEST(ShiftLeftTestSuite_check_Func_uint32, uint32_t, ShiftLeftImpl, OpType::SHIFTLEFT);
SCALAR_BINARY_FUNC_TEST(ShiftLeftTestSuite_check_Func_int16, int16_t, ShiftLeftImpl, OpType::SHIFTLEFT);
SCALAR_BINARY_FUNC_TEST(ShiftLeftTestSuite_check_Func_int32, int32_t, ShiftLeftImpl, OpType::SHIFTLEFT);

SCALAR_BINARY_FUNC_TEST(ShiftRightTestSuite_check_Func_uint16, uint16_t, ShiftRightImpl, OpType::SHIFTRIGHT);
SCALAR_BINARY_FUNC_TEST(ShiftRightTestSuite_check_Func_uint32, uint32_t, ShiftRightImpl, OpType::SHIFTRIGHT);
SCALAR_BINARY_FUNC_TEST(ShiftRightTestSuite_check_Func_int16, int16_t, ShiftRightImpl, OpType::SHIFTRIGHT);
SCALAR_BINARY_FUNC_TEST(ShiftRightTestSuite_check_Func_int32, int32_t, ShiftRightImpl, OpType::SHIFTRIGHT);

// TensorTrait Case
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_tensortrait_float, TensorTrait<float>, AddsImpl, OpType::ADDS);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_tensortrait_half, TensorTrait<half>, AddsImpl, OpType::ADDS);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_tensortrait_int16, TensorTrait<int16_t>, AddsImpl, OpType::ADDS);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_tensortrait_int32, TensorTrait<int32_t>, AddsImpl, OpType::ADDS);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_tensortrait_bfloat16, TensorTrait<bfloat16_t>, AddsImpl, OpType::ADDS);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_tensortrait_int64, TensorTrait<int64_t>, AddsImpl, OpType::ADDS);
SCALAR_BINARY_FUNC_TEST(AddsTestSuite_check_Func_tensortrait_uint64, TensorTrait<uint64_t>, AddsImpl, OpType::ADDS);

SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_tensortrait_float, TensorTrait<float>, MulsImpl, OpType::MULS);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_tensortrait_half, TensorTrait<half>, MulsImpl, OpType::MULS);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_tensortrait_int16, TensorTrait<int16_t>, MulsImpl, OpType::MULS);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_tensortrait_int32, TensorTrait<int32_t>, MulsImpl, OpType::MULS);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_tensortrait_bfloat16, TensorTrait<bfloat16_t>, MulsImpl, OpType::MULS);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_tensortrait_int64, TensorTrait<int64_t>, MulsImpl, OpType::MULS);
SCALAR_BINARY_FUNC_TEST(MulsTestSuite_check_Func_tensortrait_uint64, TensorTrait<uint64_t>, MulsImpl, OpType::MULS);

SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_tensortrait_float, TensorTrait<float>, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_tensortrait_half, TensorTrait<half>, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_tensortrait_int16, TensorTrait<int16_t>, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_tensortrait_int32, TensorTrait<int32_t>, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_tensortrait_bfloat16, TensorTrait<bfloat16_t>, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_tensortrait_int64, TensorTrait<int64_t>, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_FUNC_TEST(MaxsTestSuite_check_Func_tensortrait_uint64, TensorTrait<uint64_t>, MaxsImpl, OpType::MAXS);

SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_tensortrait_float, TensorTrait<float>, MinsImpl, OpType::MINS);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_tensortrait_half, TensorTrait<half>, MinsImpl, OpType::MINS);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_tensortrait_int16, TensorTrait<int16_t>, MinsImpl, OpType::MINS);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_tensortrait_int32, TensorTrait<int32_t>, MinsImpl, OpType::MINS);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_tensortrait_bfloat16, TensorTrait<bfloat16_t>, MinsImpl, OpType::MINS);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_tensortrait_int64, TensorTrait<int64_t>, MinsImpl, OpType::MINS);
SCALAR_BINARY_FUNC_TEST(MinsTestSuite_check_Func_tensortrait_uint64, TensorTrait<uint64_t>, MinsImpl, OpType::MINS);

SCALAR_BINARY_FUNC_TEST(ShiftLeftTestSuite_check_tensortrait_Func_uint16, TensorTrait<uint16_t>, ShiftLeftImpl, OpType::SHIFTLEFT);
SCALAR_BINARY_FUNC_TEST(ShiftLeftTestSuite_check_tensortrait_Func_uint32, TensorTrait<uint32_t>, ShiftLeftImpl, OpType::SHIFTLEFT);
SCALAR_BINARY_FUNC_TEST(ShiftLeftTestSuite_check_tensortrait_Func_int16, TensorTrait<int16_t>, ShiftLeftImpl, OpType::SHIFTLEFT);
SCALAR_BINARY_FUNC_TEST(ShiftLeftTestSuite_check_tensortrait_Func_int32, TensorTrait<int32_t>, ShiftLeftImpl, OpType::SHIFTLEFT);

SCALAR_BINARY_FUNC_TEST(ShiftRightTestSuite_checktensortrait_Func_uint16, TensorTrait<uint16_t>, ShiftRightImpl, OpType::SHIFTRIGHT);
SCALAR_BINARY_FUNC_TEST(ShiftRightTestSuite_checktensortrait_Func_uint32, TensorTrait<uint32_t>, ShiftRightImpl, OpType::SHIFTRIGHT);
SCALAR_BINARY_FUNC_TEST(ShiftRightTestSuite_checktensortrait_Func_int16, TensorTrait<int16_t>, ShiftRightImpl, OpType::SHIFTRIGHT);
SCALAR_BINARY_FUNC_TEST(ShiftRightTestSuite_checktensortrait_Func_int32, TensorTrait<int32_t>, ShiftRightImpl, OpType::SHIFTRIGHT);

SCALAR_BINARY_FUNC_TEST(LeakyReluTestSuite_check_Func_tensortrait_float, TensorTrait<float>, LeakyReluImpl, OpType::LEAKY_RELU);
SCALAR_BINARY_FUNC_TEST(LeakyReluTestSuite_check_Func_tensortrait_half, TensorTrait<half>, LeakyReluImpl, OpType::LEAKY_RELU);

template <typename SrcType, OpType OpType>
bool BinaryScalarCheckSrcValue(__ubuf__ SrcType* src)
{
    return (src[0] == static_cast<SrcType>(0));
}

bool BinaryScalarCheckCalCount(const int32_t& calCount)
{
    return (calCount == 1020);
}

#define SCALAR_BINARY_TEST(testCaseName, dataType, funcName, opType)                                   \
TEST_F(BinaryScalarTestsuite, testCaseName)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(dataType), 11, 22, 0, 3, { 3, 1, 3, 1}};                  \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType, const int32_t&))         \
        .times(1);                                                                                          \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType,                          \
        const uint64_t*, const uint8_t, const UnaryRepeatParams&))                                          \
        .times(1);                                                                                          \
    MOCKER(funcName, void(*)(__ubuf__ dataType*, __ubuf__ dataType*, dataType,                          \
        const uint64_t, const uint8_t, const UnaryRepeatParams&))                                           \
        .times(1);                                                                                          \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    BinaryScalarTest<dataType, opType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,      \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                           \
        inputParams.repeatTimes, inputParams.repeatParams);                                                 \
    EXPECT_EQ(outputGm[0], 0x00);                                                                           \
    EXPECT_EQ(outputGm[1], 0x00);                                                                           \
}                                                                                                           \

#define SCALAR_BINARY_SHIFTRIGHT_TEST(testCaseName, dataType, funcName, opType)                        \
TEST_F(BinaryScalarTestsuite, testCaseName)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(dataType), 11, 22, 0, 3, { 3, 1, 3, 1}};                  \
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
    BinaryScalarTest<dataType, opType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,      \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                           \
        inputParams.repeatTimes, inputParams.repeatParams);                                                 \
    EXPECT_EQ(outputGm[0], 0x00);                                                                           \
}

SCALAR_BINARY_TEST(AddsTestSuite_check_InputParams_float, float, AddsImpl, OpType::ADDS);
SCALAR_BINARY_TEST(AddsTestSuite_check_InputParams_half, half, AddsImpl, OpType::ADDS);
SCALAR_BINARY_TEST(AddsTestSuite_check_InputParams_int16, int16_t, AddsImpl, OpType::ADDS);
SCALAR_BINARY_TEST(AddsTestSuite_check_InputParams_int32, int32_t, AddsImpl, OpType::ADDS);

SCALAR_BINARY_TEST(MulsTestSuite_check_InputParams_float, float, MulsImpl, OpType::MULS);
SCALAR_BINARY_TEST(MulsTestSuite_check_InputParams_half, half, MulsImpl, OpType::MULS);
SCALAR_BINARY_TEST(MulsTestSuite_check_InputParams_int16, int16_t, MulsImpl, OpType::MULS);
SCALAR_BINARY_TEST(MulsTestSuite_check_InputParams_int32, int32_t, MulsImpl, OpType::MULS);

SCALAR_BINARY_TEST(MaxsTestSuite_check_InputParams_float, float, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_TEST(MaxsTestSuite_check_InputParams_half, half, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_TEST(MaxsTestSuite_check_InputParams_int16, int16_t, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_TEST(MaxsTestSuite_check_InputParams_int32, int32_t, MaxsImpl, OpType::MAXS);

SCALAR_BINARY_TEST(MinsTestSuite_check_InputParams_float, float, MinsImpl, OpType::MINS);
SCALAR_BINARY_TEST(MinsTestSuite_check_InputParams_half, half, MinsImpl, OpType::MINS);
SCALAR_BINARY_TEST(MinsTestSuite_check_InputParams_int16, int16_t, MinsImpl, OpType::MINS);
SCALAR_BINARY_TEST(MinsTestSuite_check_InputParams_int32, int32_t, MinsImpl, OpType::MINS);

SCALAR_BINARY_TEST(ShiftLeftTestSuite_check_InputParams_uint16, uint16_t, ShiftLeftImpl, OpType::SHIFTLEFT);
SCALAR_BINARY_TEST(ShiftLeftTestSuite_check_InputParams_uint32, uint32_t, ShiftLeftImpl, OpType::SHIFTLEFT);
SCALAR_BINARY_TEST(ShiftLeftTestSuite_check_InputParams_int16, int16_t, ShiftLeftImpl, OpType::SHIFTLEFT);
SCALAR_BINARY_TEST(ShiftLeftTestSuite_check_InputParams_int32, int32_t, ShiftLeftImpl, OpType::SHIFTLEFT);

SCALAR_BINARY_SHIFTRIGHT_TEST(ShiftRightTestSuite_check_InputParams_uint16, uint16_t, ShiftRightImpl, OpType::SHIFTRIGHT);
SCALAR_BINARY_SHIFTRIGHT_TEST(ShiftRightTestSuite_check_InputParams_uint32, uint32_t, ShiftRightImpl, OpType::SHIFTRIGHT);
SCALAR_BINARY_SHIFTRIGHT_TEST(ShiftRightTestSuite_check_InputParams_int16, int16_t, ShiftRightImpl, OpType::SHIFTRIGHT);
SCALAR_BINARY_SHIFTRIGHT_TEST(ShiftRightTestSuite_check_InputParams_int32, int32_t, ShiftRightImpl, OpType::SHIFTRIGHT);

template <typename SrcType>
void DataCopyUB2GMImplStub(__gm__ SrcType* dst, __ubuf__ SrcType* src, const DataCopyParams& intriParams)
{
    dst[0] = src[0];
}

template <typename SrcType>
void copy_cbuf_to_gm_align_stub(__gm__ SrcType* dst, __ubuf__ SrcType* src, uint8_t sid, uint16_t nBurst,
    uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap)
{
    dst[0] = src[0];
}

#define SCALAR_BINARY_TEST_VALUE(testCaseName, dataType, funcName, opType)                                  \
    TEST_F(BinaryScalarTestsuite, testCaseName)                                                                \
    {                                                                                                            \
        InputParams inputParams{ 1024, 1020, sizeof(dataType), 11, 22, 0, 3, { 3, 1, 3, 1 }};                   \
        dataType outputStub[inputParams.stackSize]{ 0x03 };                                                     \
        MOCKER(funcName, void (*)(__ubuf__ dataType *, __ubuf__ dataType *, dataType, const int32_t &))      \
            .times(1)                                                                                            \
            .with(outBoundP(outputStub));                                                                        \
        MOCKER(funcName, void (*)(__ubuf__ dataType *, __ubuf__ dataType *, dataType, const uint64_t *,      \
            const uint8_t, const UnaryRepeatParams &))                                                           \
            .times(1)                                                                                            \
            .with(outBoundP(outputStub));                                                                        \
        MOCKER(funcName, void (*)(__ubuf__ dataType *, __ubuf__ dataType *, dataType, const uint64_t,        \
            const uint8_t, const UnaryRepeatParams &))                                                           \
            .times(1)                                                                                            \
            .with(outBoundP(outputStub));                                                                        \
        MOCKER(DataCopyUB2GMImpl,                                                                                \
            void (*)(__gm__ dataType *, __ubuf__ dataType *, const DataCopyParams &, const uint8_t))           \
            .times(1)                                                                                            \
            .will(invoke(DataCopyUB2GMImplStub<dataType>));                                                     \
        uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize]{ 0x01 };                                 \
        uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize]{ 0x00 };                              \
        BinaryScalarTest<dataType, opType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,       \
            inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow, inputParams.repeatTimes,   \
            inputParams.repeatParams);                                                                           \
        dataType *out = reinterpret_cast<dataType *>(outputGm);                                                \
        EXPECT_EQ(out[0], static_cast<dataType>(0x03));                                                                      \
    }

#define SCALAR_BINARY_SHIFTRIGHT_TEST_VALUE(testCaseName, dataType, funcName, opType)                       \
    TEST_F(BinaryScalarTestsuite, testCaseName)                                                                \
    {                                                                                                            \
        InputParams inputParams{ 1024, 1020, sizeof(dataType), 11, 22, 0, 3, { 3, 1, 3, 1 }};                   \
        dataType outputStub[inputParams.stackSize]{ 0x03 };                                                     \
        MOCKER(funcName, void (*)(__ubuf__ dataType *, __ubuf__ dataType *, dataType, const int32_t &))      \
            .times(1)                                                                                            \
            .with(outBoundP(outputStub));                                                                        \
        MOCKER(funcName, void (*)(__ubuf__ dataType *, __ubuf__ dataType *, dataType, const uint64_t *,      \
            const uint8_t, const UnaryRepeatParams &, bool))                                                     \
            .times(1)                                                                                            \
            .with(outBoundP(outputStub));                                                                        \
        MOCKER(funcName, void (*)(__ubuf__ dataType *, __ubuf__ dataType *, dataType, const uint64_t,        \
            const uint8_t, const UnaryRepeatParams &, bool))                                                     \
            .times(1)                                                                                            \
            .with(outBoundP(outputStub));                                                                        \
        MOCKER(DataCopyUB2GMImpl,                                                                                \
            void (*)(__gm__ dataType *, __ubuf__ dataType *, const DataCopyParams &, const uint8_t))           \
            .times(1)                                                                                            \
            .will(invoke(DataCopyUB2GMImplStub<dataType>));                                                     \
        uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize]{ 0x01 };                                 \
        uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize]{ 0x00 };                              \
        BinaryScalarTest<dataType, opType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,       \
            inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow, inputParams.repeatTimes,   \
            inputParams.repeatParams);                                                                           \
        EXPECT_EQ(outputGm[0], 0x03);                                                                            \
    }

SCALAR_BINARY_TEST_VALUE(AddsTestSuite_check_Output_float, float, AddsImpl, OpType::ADDS);
SCALAR_BINARY_TEST_VALUE(AddsTestSuite_check_Output_half, half, AddsImpl, OpType::ADDS);
SCALAR_BINARY_TEST_VALUE(AddsTestSuite_check_Output_int16, int16_t, AddsImpl, OpType::ADDS);
SCALAR_BINARY_TEST_VALUE(AddsTestSuite_check_Output_int32, int32_t, AddsImpl, OpType::ADDS);

SCALAR_BINARY_TEST_VALUE(MulsTestSuite_check_Output_float, float, MulsImpl, OpType::MULS);
SCALAR_BINARY_TEST_VALUE(MulsTestSuite_check_Output_half, half, MulsImpl, OpType::MULS);
SCALAR_BINARY_TEST_VALUE(MulsTestSuite_check_Output_int16, int16_t, MulsImpl, OpType::MULS);
SCALAR_BINARY_TEST_VALUE(MulsTestSuite_check_Output_int32, int32_t, MulsImpl, OpType::MULS);

SCALAR_BINARY_TEST_VALUE(MaxsTestSuite_check_Output_float, float, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_TEST_VALUE(MaxsTestSuite_check_Output_half, half, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_TEST_VALUE(MaxsTestSuite_check_Output_int16, int16_t, MaxsImpl, OpType::MAXS);
SCALAR_BINARY_TEST_VALUE(MaxsTestSuite_check_Output_int32, int32_t, MaxsImpl, OpType::MAXS);

SCALAR_BINARY_TEST_VALUE(MinsTestSuite_check_Output_float, float, MinsImpl, OpType::MINS);
SCALAR_BINARY_TEST_VALUE(MinsTestSuite_check_Output_half, half, MinsImpl, OpType::MINS);
SCALAR_BINARY_TEST_VALUE(MinsTestSuite_check_Output_int16, int16_t, MinsImpl, OpType::MINS);
SCALAR_BINARY_TEST_VALUE(MinsTestSuite_check_Output_int32, int32_t, MinsImpl, OpType::MINS);

SCALAR_BINARY_TEST_VALUE(ShiftLeftTestSuite_check_Output_uint16, uint16_t, ShiftLeftImpl, OpType::SHIFTLEFT);
SCALAR_BINARY_TEST_VALUE(ShiftLeftTestSuite_check_Output_uint32, uint32_t, ShiftLeftImpl, OpType::SHIFTLEFT);
SCALAR_BINARY_TEST_VALUE(ShiftLeftTestSuite_check_Output_int16, int16_t, ShiftLeftImpl, OpType::SHIFTLEFT);
SCALAR_BINARY_TEST_VALUE(ShiftLeftTestSuite_check_Output_int32, int32_t, ShiftLeftImpl, OpType::SHIFTLEFT);


SCALAR_BINARY_SHIFTRIGHT_TEST_VALUE(ShiftRightTestSuite_check_Output_uint16, uint16_t, ShiftRightImpl, OpType::SHIFTRIGHT);
SCALAR_BINARY_SHIFTRIGHT_TEST_VALUE(ShiftRightTestSuite_check_Output_uint32, uint32_t, ShiftRightImpl, OpType::SHIFTRIGHT);
SCALAR_BINARY_SHIFTRIGHT_TEST_VALUE(ShiftRightTestSuite_check_Output_int16, int16_t, ShiftRightImpl, OpType::SHIFTRIGHT);
SCALAR_BINARY_SHIFTRIGHT_TEST_VALUE(ShiftRightTestSuite_check_Output_int32, int32_t, ShiftRightImpl, OpType::SHIFTRIGHT);

#define SCALAR_BINARY_TEST_UNSUPPORT(testCaseName, dataType, opType)                                    \
TEST_F(BinaryScalarTestsuite, testCaseName)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(dataType), 11, 22, 0, 3, { 3, 1, 3, 1}};                  \
    MOCKER(raise, int(*)(int)).times(3).will(returnValue(0));                                               \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    BinaryScalarTest<dataType, opType>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,      \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                           \
        inputParams.repeatTimes, inputParams.repeatParams);                                                 \
    EXPECT_EQ(outputGm[0], 0x00);                                                                           \
    EXPECT_EQ(outputGm[1], 0x00);                                                                           \
}                                                                                                           \

enum TestMode {
    ADDS = 0,
    SUBS,
    MAXS,
    MINS,
    DIVS,
    MULS,
    ANDS,
    ORS,
};

class TEST_BINARY_SCALAR_POS : public testing::Test {
protected:
    void SetUp()
    {
        SetGCoreType(2);
    }
    void TearDown()
    {
        SetGCoreType(0);
    }
};

template <typename T, int testMode>
void MainVecBinaryScalarDemo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm, __gm__ uint8_t* __restrict__ src1Gm, uint32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<T> input1Global;
    GlobalTensor<T> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    input1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    LocalTensor<T> input0Local;
    TBuffAddr tbuf;
    tbuf.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    input0Local.SetAddr(tbuf);
    input0Local.InitBuffer(0, dataSize);

    LocalTensor<T> input1Local;
    TBuffAddr tbuf1;
    tbuf1.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    input1Local.SetAddr(tbuf1);
    input1Local.InitBuffer(input0Local.GetSize() * sizeof(T), dataSize);

    LocalTensor<T> outputLocal;
    TBuffAddr tbuf3;
    tbuf3.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    outputLocal.SetAddr(tbuf3);
    outputLocal.InitBuffer(input0Local.GetSize() * sizeof(T) + input1Local.GetSize() * sizeof(T), dataSize);

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    uint8_t repeatTime = dataSize * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
    uint64_t mask[4];
    if (sizeof(T) == 1) {
        mask[0] = UINT64_MAX;
        mask[1] = UINT64_MAX;
        mask[2] = UINT64_MAX;
        mask[3] = UINT64_MAX;
    } else if (sizeof(T) == 2) {
        mask[0] = UINT64_MAX;
        mask[1] = UINT64_MAX;
    } else if (sizeof(T) == 4) {
        mask[0] = UINT64_MAX;
        mask[1] = 0;
    }

    UnaryRepeatParams repeat_params{1, 1, 8, 8};
    auto scalar = input1Local[0];
    if constexpr (testMode == int(TestMode::ADDS)) {
        Adds(outputLocal, input0Local, scalar, mask, repeatTime, repeat_params);
        Adds(outputLocal, scalar, input0Local, mask, repeatTime, repeat_params);
        Adds(outputLocal, static_cast<T>(0), input0Local, mask, repeatTime, repeat_params);
        Adds(outputLocal, input0Local, static_cast<T>(0), mask, repeatTime, repeat_params);
        Adds(outputLocal, input0Local, input0Local[0], mask, repeatTime, repeat_params);
        Adds(outputLocal, input0Local[0], input0Local, mask, repeatTime, repeat_params);
    } else if constexpr (testMode == int(TestMode::SUBS)) {
        Subs(outputLocal, input0Local, scalar, mask, repeatTime, repeat_params);
        Subs(outputLocal, scalar, input0Local, mask, repeatTime, repeat_params);
        Subs(outputLocal, input0Local, static_cast<T>(0), mask, repeatTime, repeat_params);
        Subs(outputLocal, static_cast<T>(0), input0Local, mask, repeatTime, repeat_params);
        Subs(outputLocal, input0Local, input0Local[0], mask, repeatTime, repeat_params);
        Subs(outputLocal, input0Local[0], input0Local, mask, repeatTime, repeat_params);
    } else if constexpr (testMode == int(TestMode::MAXS)) {
        Maxs(outputLocal, input0Local, scalar, mask, repeatTime, repeat_params);
        Maxs(outputLocal, scalar, input0Local, mask, repeatTime, repeat_params);
        Maxs(outputLocal, input0Local, static_cast<T>(0), mask, repeatTime, repeat_params);
        Maxs(outputLocal, static_cast<T>(0), input0Local, mask, repeatTime, repeat_params);
        Maxs(outputLocal, input0Local, input0Local[0], mask, repeatTime, repeat_params);
        Maxs(outputLocal, input0Local[0], input0Local, mask, repeatTime, repeat_params);
    } else if constexpr (testMode == int(TestMode::MINS)) {
        Mins(outputLocal, input0Local, scalar, mask, repeatTime, repeat_params);
        Mins(outputLocal, scalar, input0Local, mask, repeatTime, repeat_params);
        Mins(outputLocal, input0Local, static_cast<T>(0), mask, repeatTime, repeat_params);
        Mins(outputLocal, static_cast<T>(0), input0Local, mask, repeatTime, repeat_params);
        Mins(outputLocal, input0Local, input0Local[0], mask, repeatTime, repeat_params);
        Mins(outputLocal, input0Local[0], input0Local, mask, repeatTime, repeat_params);
    } else if constexpr (testMode == int(TestMode::MULS)) {
        Muls(outputLocal, input0Local, scalar, mask, repeatTime, repeat_params);
        Muls(outputLocal, scalar, input0Local, mask, repeatTime, repeat_params);
        Muls(outputLocal, input0Local, static_cast<T>(0), mask, repeatTime, repeat_params);
        Muls(outputLocal, static_cast<T>(0), input0Local, mask, repeatTime, repeat_params);
        Muls(outputLocal, input0Local, input0Local[0], mask, repeatTime, repeat_params);
        Muls(outputLocal, input0Local[0], input0Local, mask, repeatTime, repeat_params);
    } else if constexpr (testMode == int(TestMode::DIVS)) {
        Divs(outputLocal, input0Local, scalar, mask, repeatTime, repeat_params);
        Divs(outputLocal, scalar, input0Local, mask, repeatTime, repeat_params);
        Divs(outputLocal, input0Local, static_cast<T>(0), mask, repeatTime, repeat_params);
        Divs(outputLocal, static_cast<T>(0), input0Local, mask, repeatTime, repeat_params);
        Divs(outputLocal, input0Local, input0Local[0], mask, repeatTime, repeat_params);
        Divs(outputLocal, input0Local[0], input0Local, mask, repeatTime, repeat_params);
    } else if constexpr (testMode == int(TestMode::ANDS)) {
        Ands(outputLocal, input0Local, scalar, mask, repeatTime, repeat_params);
        Ands(outputLocal, scalar, input0Local, mask, repeatTime, repeat_params);
        Ands(outputLocal, input0Local, static_cast<T>(0), mask, repeatTime, repeat_params);
        Ands(outputLocal, static_cast<T>(0), input0Local, mask, repeatTime, repeat_params);
        Ands(outputLocal, input0Local, input0Local[0], mask, repeatTime, repeat_params);
        Ands(outputLocal, input0Local[0], input0Local, mask, repeatTime, repeat_params);
    } else if constexpr (testMode == int(TestMode::ORS)) {
        Ors(outputLocal, input0Local, scalar, mask, repeatTime, repeat_params);
        Ors(outputLocal, scalar, input0Local, mask, repeatTime, repeat_params);
        Ors(outputLocal, input0Local, static_cast<T>(0), mask, repeatTime, repeat_params);
        Ors(outputLocal, static_cast<T>(0), input0Local, mask, repeatTime, repeat_params);
        Ors(outputLocal, input0Local, input0Local[0], mask, repeatTime, repeat_params);
        Ors(outputLocal, input0Local[0], input0Local, mask, repeatTime, repeat_params);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

#define VEC_BINARY_SCALAR_POS_TESTCASE(dataType, testMode)                                              \
    TEST_F(TEST_BINARY_SCALAR_POS, BINARY_SCALAR_##dataType##_##testMode##_##Case)                                               \
    {                                                                                                               \
        uint32_t dataSize = 256;                                                                                   \
        uint8_t input0Gm[dataSize * sizeof(dataType)];                                                           \
        uint8_t input1Gm[dataSize * sizeof(dataType)];                                                           \
        uint8_t outputGm[dataSize * sizeof(dataType)];                                                           \
        MainVecBinaryScalarDemo<dataType, testMode>(outputGm, input0Gm, input1Gm, dataSize);              \
    }

// Adds
VEC_BINARY_SCALAR_POS_TESTCASE(half, 0);
VEC_BINARY_SCALAR_POS_TESTCASE(bfloat16_t, 0);
VEC_BINARY_SCALAR_POS_TESTCASE(float, 0);
VEC_BINARY_SCALAR_POS_TESTCASE(int16_t, 0);
VEC_BINARY_SCALAR_POS_TESTCASE(int32_t, 0);

// Subs
VEC_BINARY_SCALAR_POS_TESTCASE(half, 1);
VEC_BINARY_SCALAR_POS_TESTCASE(bfloat16_t, 1);
VEC_BINARY_SCALAR_POS_TESTCASE(float, 1);
VEC_BINARY_SCALAR_POS_TESTCASE(int16_t, 1);
VEC_BINARY_SCALAR_POS_TESTCASE(int32_t, 1);

// Maxs
VEC_BINARY_SCALAR_POS_TESTCASE(half, 2);
VEC_BINARY_SCALAR_POS_TESTCASE(bfloat16_t, 2);
VEC_BINARY_SCALAR_POS_TESTCASE(float, 2);
VEC_BINARY_SCALAR_POS_TESTCASE(int16_t, 2);
VEC_BINARY_SCALAR_POS_TESTCASE(int32_t, 2);

// Mins
VEC_BINARY_SCALAR_POS_TESTCASE(half, 3);
VEC_BINARY_SCALAR_POS_TESTCASE(bfloat16_t, 3);
VEC_BINARY_SCALAR_POS_TESTCASE(float, 3);
VEC_BINARY_SCALAR_POS_TESTCASE(int16_t, 3);
VEC_BINARY_SCALAR_POS_TESTCASE(int32_t, 3);

// Divs
VEC_BINARY_SCALAR_POS_TESTCASE(half, 4);
VEC_BINARY_SCALAR_POS_TESTCASE(float, 4);

// Muls
VEC_BINARY_SCALAR_POS_TESTCASE(half, 5);
VEC_BINARY_SCALAR_POS_TESTCASE(bfloat16_t, 5);
VEC_BINARY_SCALAR_POS_TESTCASE(float, 5);
VEC_BINARY_SCALAR_POS_TESTCASE(int16_t, 5);
VEC_BINARY_SCALAR_POS_TESTCASE(int32_t, 5);

// Ands
VEC_BINARY_SCALAR_POS_TESTCASE(int16_t, 6);
VEC_BINARY_SCALAR_POS_TESTCASE(uint16_t, 6);

// Ors
VEC_BINARY_SCALAR_POS_TESTCASE(int16_t, 7);
VEC_BINARY_SCALAR_POS_TESTCASE(uint16_t, 7);

template <typename T, int testMode>
void MainVecBinaryDcalarDemoWrong(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm, __gm__ uint8_t* __restrict__ src1Gm)
{
    uint32_t dataSize = 256;
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<T> input1Global;
    GlobalTensor<T> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    input1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    LocalTensor<T> input0Local;
    TBuffAddr tbuf;
    tbuf.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    input0Local.SetAddr(tbuf);
    input0Local.InitBuffer(0, dataSize);

    LocalTensor<T> input1Local;
    TBuffAddr tbuf1;
    tbuf1.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    input1Local.SetAddr(tbuf1);
    input1Local.InitBuffer(input0Local.GetSize() * sizeof(T), dataSize);

    LocalTensor<T> outputLocal;
    TBuffAddr tbuf3;
    tbuf3.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    outputLocal.SetAddr(tbuf3);
    outputLocal.InitBuffer(input0Local.GetSize() * sizeof(T) + input1Local.GetSize() * sizeof(T), dataSize);

    LocalTensor<int16_t> input0LocalS16;
    input0LocalS16.SetAddr(tbuf);
    input0LocalS16.InitBuffer(0, dataSize);

    LocalTensor<int16_t> outputLocalS16;
    outputLocalS16.SetAddr(tbuf3);
    outputLocalS16.InitBuffer(input0LocalS16.GetSize() * sizeof(int16_t) + input0LocalS16.GetSize() * sizeof(int16_t), dataSize);

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    uint8_t repeatTime = 512 * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
    uint64_t mask[4];
    if (sizeof(T) == 1) {
        mask[0] = UINT64_MAX;
        mask[1] = UINT64_MAX;
        mask[2] = UINT64_MAX;
        mask[3] = UINT64_MAX;
    } else if (sizeof(T) == 2) {
        mask[0] = UINT64_MAX;
        mask[1] = UINT64_MAX;
    } else if (sizeof(T) == 4) {
        mask[0] = UINT64_MAX;
        mask[1] = 0;
    }

    UnaryRepeatParams repeat_params{1, 1, 8, 8};
    auto scalar = input1Local[0];
    Adds(outputLocal, static_cast<T>(0), input0Local, mask, repeatTime, repeat_params);
    Subs(outputLocal, static_cast<T>(0), input0Local, mask, repeatTime, repeat_params);
    Maxs(outputLocal, static_cast<T>(0), input0Local, mask, repeatTime, repeat_params);
    Mins(outputLocal, static_cast<T>(0), input0Local, mask, repeatTime, repeat_params);
    Muls(outputLocal, static_cast<T>(0), input0Local, mask, repeatTime, repeat_params);
    Divs(outputLocal, static_cast<T>(0), input0Local, mask, repeatTime, repeat_params);
    Ands(outputLocalS16, static_cast<int16_t>(0), input0LocalS16, mask, repeatTime, repeat_params);
    Ors(outputLocalS16, static_cast<int16_t>(0), input0LocalS16, mask, repeatTime, repeat_params);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

#define VEC_BINARY_SCALAR_POS_TESTCASE_WRONG(dataType, testMode)                                                  \
    TEST_F(TEST_BINARY_SCALAR_POS, BINARY_SCALAR_WRONG_##dataType##_##testMode##_##Case)                          \
    {                                                                                                               \
        uint32_t dataSize = 256;                                                                                   \
        uint8_t input0Gm[dataSize * sizeof(dataType)];                                                           \
        uint8_t input1Gm[dataSize * sizeof(dataType)];                                                           \
        uint8_t outputGm[dataSize * sizeof(dataType)];                                                           \
        MOCKER(raise, int(*)(int)).times(8).will(returnValue(0));                                                   \
        MainVecBinaryDcalarDemoWrong<dataType, testMode>(outputGm, input0Gm, input1Gm);                   \
    }

// Adds
VEC_BINARY_SCALAR_POS_TESTCASE_WRONG(half, 0);
 