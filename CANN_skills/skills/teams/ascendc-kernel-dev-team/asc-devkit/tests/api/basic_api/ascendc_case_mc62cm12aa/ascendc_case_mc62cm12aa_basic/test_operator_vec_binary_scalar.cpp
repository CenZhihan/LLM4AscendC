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

template <typename srcType, OpType opType> class KernelBinaryScalar { // 定义kernel类函数
public:
    __aicore__ inline KernelBinaryScalar() {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, uint32_t stackSize, uint32_t dataSize,
        uint64_t maskCounter, uint64_t maskBitHigh, uint64_t maskBitLow, uint8_t repeatTimes, UnaryRepeatParams& repeatParams)
    {
        this->stackSize = stackSize; // 占用空间大小，32Bytre对齐，可能含有脏数据
        this->dataSize = dataSize; // 有效计算数据量
        this->maskCounter = maskCounter;
        this->maskBit[0] = maskBitHigh;
        this->maskBit[1] = maskBitLow;
        this->repeatTimes = repeatTimes;
        this->repeatParams = repeatParams;

        src_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType*>(src_gm), stackSize);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType*>(dst_gm), stackSize);

        pipe.InitBuffer(inQueueSrc, 1, stackSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, stackSize * sizeof(srcType));
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
        LocalTensor<srcType> srcLocal = inQueueSrc.AllocTensor<srcType>();

        DataCopy(srcLocal, src_global, stackSize);

        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        LocalTensor<srcType> srcLocal = inQueueSrc.DeQue<srcType>();

        if constexpr (opType == OpType::ADDS) {
            if constexpr (sizeof(srcType) != 8) {
                Adds(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
                Adds(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
                constexpr static BinaryConfig config = {1};
                Adds<srcType, false, config>(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            }
            Adds<srcType, true>(dstLocal, srcLocal, scalar, dataSize);
        } else if constexpr (opType == OpType::MULS) {
            if constexpr (sizeof(srcType) != 8) {
                Muls(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
                Muls(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            }
            Muls(dstLocal, srcLocal, scalar, dataSize);
        } else if constexpr (opType == OpType::MAXS) {
            if constexpr (sizeof(srcType) != 8) {
                Maxs(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
                Maxs(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            }
            Maxs(dstLocal, srcLocal, scalar, dataSize);
        } else if constexpr (opType == OpType::MINS) {
            if constexpr (sizeof(srcType) != 8) {
                Mins(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
                Mins(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            }
            Mins(dstLocal, srcLocal, scalar, dataSize);
        } else if constexpr (opType == OpType::SHIFTLEFT) {
            ShiftLeft(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
            ShiftLeft(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            ShiftLeft(dstLocal, srcLocal, scalar, dataSize);
        } else if constexpr (opType == OpType::SHIFTRIGHT) {
            ShiftRight(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams, false);
            ShiftRight(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams, false);
            ShiftRight(dstLocal, srcLocal, scalar, dataSize);
        } else {
            LeakyRelu(dstLocal, srcLocal, scalar, maskBit, repeatTimes, repeatParams);
            LeakyRelu(dstLocal, srcLocal, scalar, maskCounter, repeatTimes, repeatParams);
            LeakyRelu(dstLocal, srcLocal, scalar, dataSize);            
        }
        outQueue.EnQue<srcType>(dstLocal);

        inQueueSrc.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        DataCopy(dst_global, dstLocal, stackSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<srcType> src_global;
    GlobalTensor<srcType> dst_global;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrc;
    TQue<TPosition::VECOUT, 1> outQueue;

    uint64_t maskCounter = 0;
    uint64_t maskBit[2] = {0, 0};

    uint8_t repeatTimes = 0;
    UnaryRepeatParams repeatParams { 1, 1, 8, 8 };

    uint32_t stackSize = 0;
    uint32_t dataSize = 0;
    srcType scalar = 2;
};

template <typename srcType, OpType opType>
__aicore__ void BinaryScalarTest(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t stackSize, uint32_t dataSize,
    uint64_t maskCounter, uint64_t maskBitHigh, uint64_t maskBitLow, uint8_t repeatTimes, UnaryRepeatParams& repeatParams)
{
    KernelBinaryScalar<srcType, opType> op;
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

#define SCALAR_BINARY_FUNC_TEST(TEST_CASE_NAME, DATA_TYPE, FUNC_NAME, OP_TYPE)                              \
TEST_F(BinaryScalarTestsuite, TEST_CASE_NAME)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(DATA_TYPE), 11, 22, 0, 3, { 3, 1, 3, 1}};                  \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    BinaryScalarTest<DATA_TYPE, OP_TYPE>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,      \
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

template <typename srcType, OpType opType>
bool BinaryScalarCheckSrcValue(__ubuf__ srcType* src)
{
    return (src[0] == (srcType)0);
}

bool BinaryScalarCheckCalCount(const int32_t& calCount)
{
    return (calCount == 1020);
}

#define SCALAR_BINARY_TEST(TEST_CASE_NAME, DATA_TYPE, FUNC_NAME, OP_TYPE)                                   \
TEST_F(BinaryScalarTestsuite, TEST_CASE_NAME)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(DATA_TYPE), 11, 22, 0, 3, { 3, 1, 3, 1}};                  \
    MOCKER(FUNC_NAME, void(*)(__ubuf__ DATA_TYPE*, __ubuf__ DATA_TYPE*, DATA_TYPE, const int32_t&))         \
        .times(1);                                                                                          \
    MOCKER(FUNC_NAME, void(*)(__ubuf__ DATA_TYPE*, __ubuf__ DATA_TYPE*, DATA_TYPE,                          \
        const uint64_t*, const uint8_t, const UnaryRepeatParams&))                                          \
        .times(1);                                                                                          \
    MOCKER(FUNC_NAME, void(*)(__ubuf__ DATA_TYPE*, __ubuf__ DATA_TYPE*, DATA_TYPE,                          \
        const uint64_t, const uint8_t, const UnaryRepeatParams&))                                           \
        .times(1);                                                                                          \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    BinaryScalarTest<DATA_TYPE, OP_TYPE>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,      \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow,                           \
        inputParams.repeatTimes, inputParams.repeatParams);                                                 \
    EXPECT_EQ(outputGm[0], 0x00);                                                                           \
    EXPECT_EQ(outputGm[1], 0x00);                                                                           \
}                                                                                                           \

#define SCALAR_BINARY_SHIFTRIGHT_TEST(TEST_CASE_NAME, DATA_TYPE, FUNC_NAME, OP_TYPE)                        \
TEST_F(BinaryScalarTestsuite, TEST_CASE_NAME)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(DATA_TYPE), 11, 22, 0, 3, { 3, 1, 3, 1}};                  \
    MOCKER(FUNC_NAME, void(*)(__ubuf__ DATA_TYPE*, __ubuf__ DATA_TYPE*, DATA_TYPE, const int32_t&))         \
        .times(1);                                                                                          \
    MOCKER(FUNC_NAME, void(*)(__ubuf__ DATA_TYPE*, __ubuf__ DATA_TYPE*, DATA_TYPE,                          \
        const uint64_t*, const uint8_t, const UnaryRepeatParams&, bool))                                    \
        .times(1);                                                                                          \
    MOCKER(FUNC_NAME, void(*)(__ubuf__ DATA_TYPE*, __ubuf__ DATA_TYPE*, DATA_TYPE,                          \
        const uint64_t, const uint8_t, const UnaryRepeatParams&, bool))                                     \
        .times(1);                                                                                          \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    BinaryScalarTest<DATA_TYPE, OP_TYPE>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,      \
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

template <typename srcType>
void DataCopyUB2GMImplStub(__gm__ srcType* dst, __ubuf__ srcType* src, const DataCopyParams& intriParams)
{
    dst[0] = src[0];
}

template <typename srcType>
void copy_cbuf_to_gm_align_stub(__gm__ srcType* dst, __ubuf__ srcType* src, uint8_t sid, uint16_t nBurst,
    uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap)
{
    dst[0] = src[0];
}

#define SCALAR_BINARY_TEST_VALUE(TEST_CASE_NAME, DATA_TYPE, FUNC_NAME, OP_TYPE)                                  \
    TEST_F(BinaryScalarTestsuite, TEST_CASE_NAME)                                                                \
    {                                                                                                            \
        InputParams inputParams{ 1024, 1020, sizeof(DATA_TYPE), 11, 22, 0, 3, { 3, 1, 3, 1 }};                   \
        DATA_TYPE outputStub[inputParams.stackSize]{ 0x03 };                                                     \
        MOCKER(FUNC_NAME, void (*)(__ubuf__ DATA_TYPE *, __ubuf__ DATA_TYPE *, DATA_TYPE, const int32_t &))      \
            .times(1)                                                                                            \
            .with(outBoundP(outputStub));                                                                        \
        MOCKER(FUNC_NAME, void (*)(__ubuf__ DATA_TYPE *, __ubuf__ DATA_TYPE *, DATA_TYPE, const uint64_t *,      \
            const uint8_t, const UnaryRepeatParams &))                                                           \
            .times(1)                                                                                            \
            .with(outBoundP(outputStub));                                                                        \
        MOCKER(FUNC_NAME, void (*)(__ubuf__ DATA_TYPE *, __ubuf__ DATA_TYPE *, DATA_TYPE, const uint64_t,        \
            const uint8_t, const UnaryRepeatParams &))                                                           \
            .times(1)                                                                                            \
            .with(outBoundP(outputStub));                                                                        \
        MOCKER(DataCopyUB2GMImpl,                                                                                \
            void (*)(__gm__ DATA_TYPE *, __ubuf__ DATA_TYPE *, const DataCopyParams &, const uint8_t))           \
            .times(1)                                                                                            \
            .will(invoke(DataCopyUB2GMImplStub<DATA_TYPE>));                                                     \
        uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize]{ 0x01 };                                 \
        uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize]{ 0x00 };                              \
        BinaryScalarTest<DATA_TYPE, OP_TYPE>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,       \
            inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow, inputParams.repeatTimes,   \
            inputParams.repeatParams);                                                                           \
        DATA_TYPE *out = reinterpret_cast<DATA_TYPE *>(outputGm);                                                \
        EXPECT_EQ(out[0], (DATA_TYPE)0x03);                                                                      \
    }

#define SCALAR_BINARY_SHIFTRIGHT_TEST_VALUE(TEST_CASE_NAME, DATA_TYPE, FUNC_NAME, OP_TYPE)                       \
    TEST_F(BinaryScalarTestsuite, TEST_CASE_NAME)                                                                \
    {                                                                                                            \
        InputParams inputParams{ 1024, 1020, sizeof(DATA_TYPE), 11, 22, 0, 3, { 3, 1, 3, 1 }};                   \
        DATA_TYPE outputStub[inputParams.stackSize]{ 0x03 };                                                     \
        MOCKER(FUNC_NAME, void (*)(__ubuf__ DATA_TYPE *, __ubuf__ DATA_TYPE *, DATA_TYPE, const int32_t &))      \
            .times(1)                                                                                            \
            .with(outBoundP(outputStub));                                                                        \
        MOCKER(FUNC_NAME, void (*)(__ubuf__ DATA_TYPE *, __ubuf__ DATA_TYPE *, DATA_TYPE, const uint64_t *,      \
            const uint8_t, const UnaryRepeatParams &, bool))                                                     \
            .times(1)                                                                                            \
            .with(outBoundP(outputStub));                                                                        \
        MOCKER(FUNC_NAME, void (*)(__ubuf__ DATA_TYPE *, __ubuf__ DATA_TYPE *, DATA_TYPE, const uint64_t,        \
            const uint8_t, const UnaryRepeatParams &, bool))                                                     \
            .times(1)                                                                                            \
            .with(outBoundP(outputStub));                                                                        \
        MOCKER(DataCopyUB2GMImpl,                                                                                \
            void (*)(__gm__ DATA_TYPE *, __ubuf__ DATA_TYPE *, const DataCopyParams &, const uint8_t))           \
            .times(1)                                                                                            \
            .will(invoke(DataCopyUB2GMImplStub<DATA_TYPE>));                                                     \
        uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize]{ 0x01 };                                 \
        uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize]{ 0x00 };                              \
        BinaryScalarTest<DATA_TYPE, OP_TYPE>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,       \
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

#define SCALAR_BINARY_TEST_UNSUPPORT(TEST_CASE_NAME, DATA_TYPE, OP_TYPE)                                    \
TEST_F(BinaryScalarTestsuite, TEST_CASE_NAME)                                                               \
{                                                                                                           \
    InputParams inputParams{1024,  1020,  sizeof(DATA_TYPE), 11, 22, 0, 3, { 3, 1, 3, 1}};                  \
    MOCKER(raise, int(*)(int)).times(3).will(returnValue(0));                                               \
    uint8_t srcGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                               \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    BinaryScalarTest<DATA_TYPE, OP_TYPE>(srcGm, outputGm, inputParams.stackSize, inputParams.dataSize,      \
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

template <typename T, int TEST_MODE>
void main_vec_binary_scalar_demo(__gm__ uint8_t* __restrict__ dst_gm, __gm__ uint8_t* __restrict__ src0_gm, __gm__ uint8_t* __restrict__ src1_gm, uint32_t data_size)
{
    TPipe tpipe;
    GlobalTensor<T> input0_global;
    GlobalTensor<T> input1_global;
    GlobalTensor<T> output_global;
    input0_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0_gm), data_size);
    input1_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1_gm), data_size);
    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst_gm), data_size);

    LocalTensor<T> input0_local;
    TBuffAddr tbuf;
    tbuf.logicPos = (uint8_t)TPosition::VECCALC;
    input0_local.SetAddr(tbuf);
    input0_local.InitBuffer(0, data_size);

    LocalTensor<T> input1_local;
    TBuffAddr tbuf1;
    tbuf1.logicPos = (uint8_t)TPosition::VECCALC;
    input1_local.SetAddr(tbuf1);
    input1_local.InitBuffer(input0_local.GetSize() * sizeof(T), data_size);

    LocalTensor<T> output_local;
    TBuffAddr tbuf3;
    tbuf3.logicPos = (uint8_t)TPosition::VECCALC;
    output_local.SetAddr(tbuf3);
    output_local.InitBuffer(input0_local.GetSize() * sizeof(T) + input1_local.GetSize() * sizeof(T), data_size);

    DataCopy(input0_local, input0_global, data_size);
    DataCopy(input1_local, input1_global, data_size);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
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
    auto scalar = input1_local[0];
    if constexpr (TEST_MODE == int(TestMode::ADDS)) {
        Adds(output_local, input0_local, scalar, mask, repeatTime, repeat_params);
        Adds(output_local, scalar, input0_local, mask, repeatTime, repeat_params);
        Adds(output_local, (T)0, input0_local, mask, repeatTime, repeat_params);
        Adds(output_local, input0_local, (T)0, mask, repeatTime, repeat_params);
        Adds(output_local, input0_local, input0_local[0], mask, repeatTime, repeat_params);
        Adds(output_local, input0_local[0], input0_local, mask, repeatTime, repeat_params);
    } else if constexpr (TEST_MODE == int(TestMode::SUBS)) {
        Subs(output_local, input0_local, scalar, mask, repeatTime, repeat_params);
        Subs(output_local, scalar, input0_local, mask, repeatTime, repeat_params);
        Subs(output_local, input0_local, (T)0, mask, repeatTime, repeat_params);
        Subs(output_local, (T)0, input0_local, mask, repeatTime, repeat_params);
        Subs(output_local, input0_local, input0_local[0], mask, repeatTime, repeat_params);
        Subs(output_local, input0_local[0], input0_local, mask, repeatTime, repeat_params);
    } else if constexpr (TEST_MODE == int(TestMode::MAXS)) {
        Maxs(output_local, input0_local, scalar, mask, repeatTime, repeat_params);
        Maxs(output_local, scalar, input0_local, mask, repeatTime, repeat_params);
        Maxs(output_local, input0_local, (T)0, mask, repeatTime, repeat_params);
        Maxs(output_local, (T)0, input0_local, mask, repeatTime, repeat_params);
        Maxs(output_local, input0_local, input0_local[0], mask, repeatTime, repeat_params);
        Maxs(output_local, input0_local[0], input0_local, mask, repeatTime, repeat_params);
    } else if constexpr (TEST_MODE == int(TestMode::MINS)) {
        Mins(output_local, input0_local, scalar, mask, repeatTime, repeat_params);
        Mins(output_local, scalar, input0_local, mask, repeatTime, repeat_params);
        Mins(output_local, input0_local, (T)0, mask, repeatTime, repeat_params);
        Mins(output_local, (T)0, input0_local, mask, repeatTime, repeat_params);
        Mins(output_local, input0_local, input0_local[0], mask, repeatTime, repeat_params);
        Mins(output_local, input0_local[0], input0_local, mask, repeatTime, repeat_params);
    } else if constexpr (TEST_MODE == int(TestMode::MULS)) {
        Muls(output_local, input0_local, scalar, mask, repeatTime, repeat_params);
        Muls(output_local, scalar, input0_local, mask, repeatTime, repeat_params);
        Muls(output_local, input0_local, (T)0, mask, repeatTime, repeat_params);
        Muls(output_local, (T)0, input0_local, mask, repeatTime, repeat_params);
        Muls(output_local, input0_local, input0_local[0], mask, repeatTime, repeat_params);
        Muls(output_local, input0_local[0], input0_local, mask, repeatTime, repeat_params);
    } else if constexpr (TEST_MODE == int(TestMode::DIVS)) {
        Divs(output_local, input0_local, scalar, mask, repeatTime, repeat_params);
        Divs(output_local, scalar, input0_local, mask, repeatTime, repeat_params);
        Divs(output_local, input0_local, (T)0, mask, repeatTime, repeat_params);
        Divs(output_local, (T)0, input0_local, mask, repeatTime, repeat_params);
        Divs(output_local, input0_local, input0_local[0], mask, repeatTime, repeat_params);
        Divs(output_local, input0_local[0], input0_local, mask, repeatTime, repeat_params);
    } else if constexpr (TEST_MODE == int(TestMode::ANDS)) {
        Ands(output_local, input0_local, scalar, mask, repeatTime, repeat_params);
        Ands(output_local, scalar, input0_local, mask, repeatTime, repeat_params);
        Ands(output_local, input0_local, (T)0, mask, repeatTime, repeat_params);
        Ands(output_local, (T)0, input0_local, mask, repeatTime, repeat_params);
        Ands(output_local, input0_local, input0_local[0], mask, repeatTime, repeat_params);
        Ands(output_local, input0_local[0], input0_local, mask, repeatTime, repeat_params);
    } else if constexpr (TEST_MODE == int(TestMode::ORS)) {
        Ors(output_local, input0_local, scalar, mask, repeatTime, repeat_params);
        Ors(output_local, scalar, input0_local, mask, repeatTime, repeat_params);
        Ors(output_local, input0_local, (T)0, mask, repeatTime, repeat_params);
        Ors(output_local, (T)0, input0_local, mask, repeatTime, repeat_params);
        Ors(output_local, input0_local, input0_local[0], mask, repeatTime, repeat_params);
        Ors(output_local, input0_local[0], input0_local, mask, repeatTime, repeat_params);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(output_global, output_local, data_size);
    pipe_barrier(PIPE_ALL);
}

#define VEC_BINARY_SCALAR_POS_TESTCASE(DATA_TYPE, TEST_MODE)                                              \
    TEST_F(TEST_BINARY_SCALAR_POS, BINARY_SCALAR_##DATA_TYPE##_##TEST_MODE##_##Case)                                               \
    {                                                                                                               \
        uint32_t data_size = 256;                                                                                   \
        uint8_t input0_gm[data_size * sizeof(DATA_TYPE)];                                                           \
        uint8_t input1_gm[data_size * sizeof(DATA_TYPE)];                                                           \
        uint8_t output_gm[data_size * sizeof(DATA_TYPE)];                                                           \
        main_vec_binary_scalar_demo<DATA_TYPE, TEST_MODE>(output_gm, input0_gm, input1_gm, data_size);              \
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

template <typename T, int TEST_MODE>
void main_vec_binary_scalar_demo_wrong(__gm__ uint8_t* __restrict__ dst_gm, __gm__ uint8_t* __restrict__ src0_gm, __gm__ uint8_t* __restrict__ src1_gm)
{
    uint32_t data_size = 256;
    TPipe tpipe;
    GlobalTensor<T> input0_global;
    GlobalTensor<T> input1_global;
    GlobalTensor<T> output_global;
    input0_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0_gm), data_size);
    input1_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1_gm), data_size);
    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst_gm), data_size);

    LocalTensor<T> input0_local;
    TBuffAddr tbuf;
    tbuf.logicPos = (uint8_t)TPosition::VECCALC;
    input0_local.SetAddr(tbuf);
    input0_local.InitBuffer(0, data_size);

    LocalTensor<T> input1_local;
    TBuffAddr tbuf1;
    tbuf1.logicPos = (uint8_t)TPosition::VECCALC;
    input1_local.SetAddr(tbuf1);
    input1_local.InitBuffer(input0_local.GetSize() * sizeof(T), data_size);

    LocalTensor<T> output_local;
    TBuffAddr tbuf3;
    tbuf3.logicPos = (uint8_t)TPosition::VECCALC;
    output_local.SetAddr(tbuf3);
    output_local.InitBuffer(input0_local.GetSize() * sizeof(T) + input1_local.GetSize() * sizeof(T), data_size);

    LocalTensor<int16_t> input0_local_s16;
    input0_local_s16.SetAddr(tbuf);
    input0_local_s16.InitBuffer(0, data_size);

    LocalTensor<int16_t> output_local_s16;
    output_local_s16.SetAddr(tbuf3);
    output_local_s16.InitBuffer(input0_local_s16.GetSize() * sizeof(int16_t) + input0_local_s16.GetSize() * sizeof(int16_t), data_size);

    DataCopy(input0_local, input0_global, data_size);
    DataCopy(input1_local, input1_global, data_size);

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
    auto scalar = input1_local[0];
    Adds(output_local, (T)0, input0_local, mask, repeatTime, repeat_params);
    Subs(output_local, (T)0, input0_local, mask, repeatTime, repeat_params);
    Maxs(output_local, (T)0, input0_local, mask, repeatTime, repeat_params);
    Mins(output_local, (T)0, input0_local, mask, repeatTime, repeat_params);
    Muls(output_local, (T)0, input0_local, mask, repeatTime, repeat_params);
    Divs(output_local, (T)0, input0_local, mask, repeatTime, repeat_params);
    Ands(output_local_s16, (int16_t)0, input0_local_s16, mask, repeatTime, repeat_params);
    Ors(output_local_s16, (int16_t)0, input0_local_s16, mask, repeatTime, repeat_params);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(output_global, output_local, data_size);
    pipe_barrier(PIPE_ALL);
}

#define VEC_BINARY_SCALAR_POS_TESTCASE_WRONG(DATA_TYPE, TEST_MODE)                                                  \
    TEST_F(TEST_BINARY_SCALAR_POS, BINARY_SCALAR_WRONG_##DATA_TYPE##_##TEST_MODE##_##Case)                          \
    {                                                                                                               \
        uint32_t data_size = 256;                                                                                   \
        uint8_t input0_gm[data_size * sizeof(DATA_TYPE)];                                                           \
        uint8_t input1_gm[data_size * sizeof(DATA_TYPE)];                                                           \
        uint8_t output_gm[data_size * sizeof(DATA_TYPE)];                                                           \
        MOCKER(raise, int(*)(int)).times(8).will(returnValue(0));                                                   \
        main_vec_binary_scalar_demo_wrong<DATA_TYPE, TEST_MODE>(output_gm, input0_gm, input1_gm);                   \
    }

// Adds
VEC_BINARY_SCALAR_POS_TESTCASE_WRONG(half, 0);
 