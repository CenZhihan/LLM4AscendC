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
    Neg = 0,
    Relu = 1,
    Not = 2,
    Reciprocal = 3
};

template <typename SrcType, OpType CmdKey> class KernelVecUnaryB64 {
public:
    __aicore__ inline KernelVecUnaryB64() {}
    __aicore__ inline void Init(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t stackSize, uint32_t dataSize,
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

        src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ SrcType*>(src0Gm), stackSize);
        src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ SrcType*>(src1Gm), stackSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ SrcType*>(dstGm), stackSize);

        pipe.InitBuffer(inQueueSrc0, 1, stackSize * sizeof(SrcType));
        pipe.InitBuffer(inQueueSrc1, 1, stackSize * sizeof(SrcType));
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
        LocalTensor<SrcType> src0Local = inQueueSrc0.AllocTensor<SrcType>();
        LocalTensor<SrcType> src1Local = inQueueSrc1.AllocTensor<SrcType>();

        DataCopy(src0Local, src0Global, stackSize);
        DataCopy(src1Local, src1Global, stackSize);

        inQueueSrc0.EnQue(src0Local);
        inQueueSrc1.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<SrcType> dstLocal = outQueue.AllocTensor<SrcType>();
        LocalTensor<SrcType> src0Local = inQueueSrc0.DeQue<SrcType>();
        LocalTensor<SrcType> src1Local = inQueueSrc1.DeQue<SrcType>();

        if constexpr (CmdKey == OpType::Neg) {
            Neg(dstLocal, src0Local, dataSize);
        } else if constexpr (CmdKey == OpType::Relu) {
            Relu(dstLocal, src0Local, dataSize);
        } else if constexpr (CmdKey == OpType::Not) {
            Not(dstLocal, src0Local, dataSize);
        } else if constexpr (CmdKey == OpType::Reciprocal) {
            Reciprocal(dstLocal, src0Local, dataSize);
        }

        outQueue.EnQue<SrcType>(dstLocal);

        inQueueSrc0.FreeTensor(src0Local);
        inQueueSrc1.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<SrcType> dstLocal = outQueue.DeQue<SrcType>();
        DataCopy(dstGlobal, dstLocal, stackSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<SrcType> src0Global;
    GlobalTensor<SrcType> src1Global;
    GlobalTensor<SrcType> dstGlobal;

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

template <typename SrcType, OpType CmdKey>
__aicore__ void MainVecUnaryTest(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t stackSize,
    uint32_t dataSize, uint64_t maskCounter, uint64_t maskBitHigh, uint64_t maskBitLow, uint8_t repeatTimes,
    BinaryRepeatParams& repeatParams)
{
    KernelVecUnaryB64<SrcType, CmdKey> op;
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

class VecUnaryB64TestSuite : public ::testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "VecUnaryTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "VecUnaryTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {GlobalMockObject::verify();}
};

// no mocker test, for coverage
#define REGISTER_VEC_UNARY_NO_MOCKER_B64(cmdKey, cmdKeyLower, dtype, dtypeKey, dataLength)                     \
TEST_F(VecUnaryB64TestSuite, VecUnaryB64TestSuite##_##cmdKey##_##dtype##_no_mocker)                             \
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
    MainVecUnaryTest<dtype, OpType::cmdKey>(src0Gm, src1Gm, outputGm, inputParams.stackSize, inputParams.dataSize,      \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow, inputParams.repeatTimes,  \
        inputParams.repeatParams);                                                          \
}

REGISTER_VEC_UNARY_NO_MOCKER_B64(Neg, neg, int64_t, s64, 256)
REGISTER_VEC_UNARY_NO_MOCKER_B64(Relu, relu, int64_t, s64, 256)
REGISTER_VEC_UNARY_NO_MOCKER_B64(Not, not, int64_t, s64, 256)
REGISTER_VEC_UNARY_NO_MOCKER_B64(Not, not, uint64_t, u64, 256)
REGISTER_VEC_UNARY_NO_MOCKER_B64(Reciprocal, rec, int64_t, s64, 256)
REGISTER_VEC_UNARY_NO_MOCKER_B64(Reciprocal, rec, uint64_t, u64, 256)