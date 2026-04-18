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
    Add = 0,
    Sub = 1,
    Mul = 2,
    Max = 3,
    Min = 4,
    Div = 5,
    Or = 6,
    And = 7,
    Madd = 8
};

template <typename srcType, OpType cmdKey> class KernelVecBinaryB64 {
public:
    __aicore__ inline KernelVecBinaryB64() {}
    __aicore__ inline void Init(GM_ADDR src0_gm, GM_ADDR src1_gm, GM_ADDR dst_gm, uint32_t stackSize, uint32_t dataSize,
        uint64_t maskCounter, uint64_t maskBitHigh, uint64_t maskBitLow, uint8_t repeatTimes,
        BinaryRepeatParams& repeatParams)
    {
        this->stackSize = stackSize;            // 占用空间大小，32Byte对齐，可能含有脏数据
        this->dataSize = dataSize;              // 有效计算数据量
        this->maskCounter = maskCounter;
        this->maskBit[0] = maskBitHigh;
        this->maskBit[1] = maskBitLow;
        this->repeatTimes = repeatTimes;
        this->repeatParams = repeatParams;

        src0_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType*>(src0_gm), stackSize);
        src1_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType*>(src1_gm), stackSize);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType*>(dst_gm), stackSize);

        pipe.InitBuffer(inQueueSrc0, 1, stackSize * sizeof(srcType));
        pipe.InitBuffer(inQueueSrc1, 1, stackSize * sizeof(srcType));
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
        LocalTensor<srcType> src0Local = inQueueSrc0.AllocTensor<srcType>();
        LocalTensor<srcType> src1Local = inQueueSrc1.AllocTensor<srcType>();

        DataCopy(src0Local, src0_global, stackSize);
        DataCopy(src1Local, src1_global, stackSize);

        inQueueSrc0.EnQue(src0Local);
        inQueueSrc1.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        LocalTensor<srcType> src0Local = inQueueSrc0.DeQue<srcType>();
        LocalTensor<srcType> src1Local = inQueueSrc1.DeQue<srcType>();

        if (cmdKey == OpType::Add) {
            Add(dstLocal, src0Local, src1Local, dataSize);
        } else if (cmdKey == OpType::Sub) {
            Sub(dstLocal, src0Local, src1Local, dataSize);
        } else if (cmdKey == OpType::Mul) {
            Mul(dstLocal, src0Local, src1Local, dataSize);
        } else if (cmdKey == OpType::Max) {
            Max(dstLocal, src0Local, src1Local, dataSize);
        } else if (cmdKey == OpType::Min) {
            Min(dstLocal, src0Local, src1Local, dataSize);
        } else if (cmdKey == OpType::Div) {
            Div(dstLocal, src0Local, src1Local, dataSize);
        } else if (cmdKey == OpType::Or) {
            Or(dstLocal, src0Local, src1Local, dataSize);
        } else if (cmdKey == OpType::And) {
            And(dstLocal, src0Local, src1Local, dataSize);
        } 

        outQueue.EnQue<srcType>(dstLocal);

        inQueueSrc0.FreeTensor(src0Local);
        inQueueSrc1.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        DataCopy(dst_global, dstLocal, stackSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<srcType> src0_global;
    GlobalTensor<srcType> src1_global;
    GlobalTensor<srcType> dst_global;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrc0;
    TQue<TPosition::VECIN, 1> inQueueSrc1;
    TQue<TPosition::VECOUT, 1> outQueue;

    uint64_t maskCounter = 0;
    uint64_t maskBit[2] = {0, 0};

    uint8_t repeatTimes = 0;
    BinaryRepeatParams repeatParams { 1, 1, 1, 8, 8, 8 };

    uint32_t stackSize = 0;
    uint32_t dataSize = 0;
};

template <typename srcType, OpType cmdKey>
__aicore__ void main_Vec_Binary_test(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t stackSize,
    uint32_t dataSize, uint64_t maskCounter, uint64_t maskBitHigh, uint64_t maskBitLow, uint8_t repeatTimes,
    BinaryRepeatParams& repeatParams)
{
    KernelVecBinaryB64<srcType, cmdKey> op;
    op.Init(src0Gm, src1Gm, dstGm, stackSize, dataSize, maskCounter, maskBitHigh, maskBitLow, repeatTimes, repeatParams);
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
    BinaryRepeatParams repeatParams;
};

class VecBinaryB64TestSuite : public ::testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "VecBinaryTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "VecBinaryTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {GlobalMockObject::verify();}
};

// no mocker test, for coverage
#define REGISTER_VEC_BINARY_NO_MOCKER_B64(cmdKey, cmdKeyLower, dtype, dtypeKey, dataLength)                     \
TEST_F(VecBinaryB64TestSuite, VecBinaryB64TestSuite##_##cmdKey##_##dtype##_no_mocker)                             \
{                                                                                                           \
    uint64_t mask = 256 / sizeof(dtype);                                                                    \
    uint64_t maskHigh = 0xffffffff;                                                                         \
    uint64_t maskLow = 0x000000000;                                                                         \
    uint8_t repeatTimes = dataLength / (256 / sizeof(dtype));                                               \
    InputParams inputParams{dataLength, dataLength, sizeof(dtype), mask, maskHigh, maskLow, repeatTimes,    \
        {1, 1, 1, 8, 8, 8}};                                                                                \
    uint8_t src0Gm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                              \
    uint8_t src1Gm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                              \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    main_Vec_Binary_test<dtype, OpType::cmdKey>(src0Gm, src1Gm, outputGm, inputParams.stackSize, inputParams.dataSize,      \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow, inputParams.repeatTimes,  \
        inputParams.repeatParams);                                                          \
}

REGISTER_VEC_BINARY_NO_MOCKER_B64(And, and, int64_t, s64, 256)
REGISTER_VEC_BINARY_NO_MOCKER_B64(Or, or, int64_t, s64, 256)
REGISTER_VEC_BINARY_NO_MOCKER_B64(Max, max, int64_t, s64, 256)
REGISTER_VEC_BINARY_NO_MOCKER_B64(Min, min, int64_t, s64, 256)
REGISTER_VEC_BINARY_NO_MOCKER_B64(And, and, uint64_t, u64, 256)
REGISTER_VEC_BINARY_NO_MOCKER_B64(Or, or, uint64_t, u64, 256)
REGISTER_VEC_BINARY_NO_MOCKER_B64(Max, max, uint64_t, u64, 256)
REGISTER_VEC_BINARY_NO_MOCKER_B64(Min, min, uint64_t, u64, 256)