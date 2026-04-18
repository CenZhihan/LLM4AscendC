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
    Exp = 0,
    Ln = 1,
    Abs = 2,
    Reciprocal = 3,
    Sqrt = 4,
    Rsqrt = 5,
    Not = 6,
    Relu = 7
};

constexpr RsqrtConfig Rsqrtconfig = { RsqrtAlgo::FAST_INVERSE };
constexpr SqrtConfig Sqrtconfig = { SqrtAlgo::FAST_INVERSE };
template <typename srcType, OpType cmdKey> class KernelVecUnary {
public:
    __aicore__ inline KernelVecUnary() {}

    __aicore__ inline void Init(GM_ADDR src0_gm, GM_ADDR dst_gm, uint32_t stackSize, uint32_t dataSize,
        uint64_t maskCounter, uint64_t maskBitHigh, uint64_t maskBitLow, uint8_t repeatTimes,
        UnaryRepeatParams& repeatParams)
    {
        this->stackSize = stackSize;            // 占用空间大小，32Byte对齐，可能含有脏数据
        this->dataSize = dataSize;              // 有效计算数据量
        this->maskCounter = maskCounter;
        this->maskBit[0] = maskBitHigh;
        this->maskBit[1] = maskBitLow;
        this->repeatTimes = repeatTimes;
        this->repeatParams = repeatParams;

        src0_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType*>(src0_gm), stackSize);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType*>(dst_gm), stackSize);

        pipe.InitBuffer(inQueueSrc0, 1, stackSize * sizeof(srcType));
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
        DataCopy(src0Local, src0_global, stackSize);
        inQueueSrc0.EnQue(src0Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        LocalTensor<srcType> src0Local = inQueueSrc0.DeQue<srcType>();

        if constexpr (cmdKey == OpType::Exp) {
            Exp(dstLocal, src0Local, maskCounter, repeatTimes, repeatParams);
            Exp(dstLocal, src0Local, maskBit, repeatTimes, repeatParams);
            Exp(dstLocal, src0Local, dataSize);
        } else if constexpr (cmdKey == OpType::Ln) {
            Ln(dstLocal, src0Local, maskCounter, repeatTimes, repeatParams);
            Ln(dstLocal, src0Local, maskBit, repeatTimes, repeatParams);
            Ln(dstLocal, src0Local, dataSize);
        } else if constexpr (cmdKey == OpType::Abs) {
            Abs(dstLocal, src0Local, maskCounter, repeatTimes, repeatParams);
            Abs(dstLocal, src0Local, maskBit, repeatTimes, repeatParams);
            Abs(dstLocal, src0Local, dataSize);
        } else if constexpr (cmdKey == OpType::Reciprocal) {
            Reciprocal(dstLocal, src0Local, maskCounter, repeatTimes, repeatParams);
            Reciprocal(dstLocal, src0Local, maskBit, repeatTimes, repeatParams);
            Reciprocal(dstLocal, src0Local, dataSize);
        } else if constexpr (cmdKey == OpType::Sqrt) {
            Sqrt(dstLocal, src0Local, maskCounter, repeatTimes, repeatParams);
            Sqrt(dstLocal, src0Local, maskBit, repeatTimes, repeatParams);
            Sqrt(dstLocal, src0Local, dataSize);
            if constexpr (std::is_same<srcType, float>::value) {
                Sqrt<srcType, Sqrtconfig>(dstLocal, src0Local, dataSize);
            }
        } else if constexpr (cmdKey == OpType::Rsqrt) {
            Rsqrt(dstLocal, src0Local, maskCounter, repeatTimes, repeatParams);
            Rsqrt(dstLocal, src0Local, maskBit, repeatTimes, repeatParams);
            Rsqrt(dstLocal, src0Local, dataSize);
            if constexpr (std::is_same<srcType, float>::value) {
                Rsqrt<srcType, Rsqrtconfig>(dstLocal, src0Local, dataSize);
            }
        } else if constexpr (cmdKey == OpType::Not) {
            Not(dstLocal, src0Local, maskCounter, repeatTimes, repeatParams);
            Not(dstLocal, src0Local, maskBit, repeatTimes, repeatParams);
            Not(dstLocal, src0Local, dataSize);
        } else if constexpr (cmdKey == OpType::Relu) {
            Relu(dstLocal, src0Local, maskCounter, repeatTimes, repeatParams);
            Relu(dstLocal, src0Local, maskBit, repeatTimes, repeatParams);
            Relu(dstLocal, src0Local, dataSize);
        }

        outQueue.EnQue<srcType>(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        DataCopy(dst_global, dstLocal, stackSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<srcType> src0_global;
    GlobalTensor<srcType> dst_global;

    TPipe pipe;
    TQue<TPosition::VECIN, 1>  inQueueSrc0;
    TQue<TPosition::VECOUT, 1> outQueue;

    uint64_t maskCounter = 0;
    uint64_t maskBit[2] = {0, 0};

    uint8_t repeatTimes = 0;
    UnaryRepeatParams repeatParams {1, 1, 8, 8};

    uint32_t stackSize = 0;
    uint32_t dataSize = 0;
};

template <typename srcType, OpType cmdKey>
__aicore__ void main_Vec_Unary_test(GM_ADDR src0Gm, GM_ADDR dstGm, uint32_t stackSize, uint32_t dataSize,
    uint64_t maskCounter, uint64_t maskBitHigh, uint64_t maskBitLow, uint8_t repeatTimes,
    UnaryRepeatParams& repeatParams)
{
    KernelVecUnary<srcType, cmdKey> op;
    op.Init(src0Gm, dstGm, stackSize, dataSize, maskCounter, maskBitHigh, maskBitLow, repeatTimes, repeatParams);
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

class VecUnaryTestSuite : public ::testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "VecUnaryTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "VecUnaryTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {GlobalMockObject::verify();}
};


// XXXImpl function: Check params other than tensors
template <typename srcType>
bool UnaryImplCheckMask(const uint64_t mask) {
    return (mask == 256 / sizeof(srcType));
}

template <typename srcType>
bool UnaryImplCheckMask2(const uint64_t mask[2]) {
    return (mask[0] == 0xffffffff && mask[1] == 0x000000000);
}

template <typename srcType>
bool UnaryImplCheckRepeatTimes(const uint8_t repeatTimes) {
    return (repeatTimes == 256 / (256 / sizeof(srcType)));
}

bool UnaryImplCheckUnaryRepeatParams(const UnaryRepeatParams& repeatParams) {
    return repeatParams.srcBlkStride == 1 && repeatParams.dstBlkStride == 1 && repeatParams.srcRepStride == 8 &&
        repeatParams.dstRepStride == 8;
}

bool UnaryImplCheckCalCount(const uint32_t calCount) {
    return (calCount == 256);
}

template <typename srcType>
void UnaryDataCopyUB2GMImplStub(__gm__ srcType* dst, __ubuf__ srcType* src, const DataCopyParams& intriParams) {
    dst[0] = src[0];
}

// Complete process: Exp -> ExpImpl -> vadd        therefore check how many times vadd has been launched
// cmdKey: Exp, cmdKeyLower: add, dtypeKey: s16
#define REGISTER_VEC_UNARY(cmdKey, cmdKeyLower, dtype, dtypeKey, dataLength)                                \
TEST_F(VecUnaryTestSuite, VecUnaryTestSuite##_##cmdKey##_##dtype)                                           \
{                                                                                                           \
    uint64_t mask = 256 / sizeof(dtype);                                                                    \
    uint64_t maskHigh = 0xffffffff;                                                                         \
    uint64_t maskLow = 0x000000000;                                                                         \
    uint8_t repeatTimes = dataLength / (256 / sizeof(dtype));                                               \
    InputParams inputParams{dataLength, dataLength, sizeof(dtype), mask, maskHigh, maskLow, repeatTimes,    \
        {1, 1, 8, 8}};                                                                                      \
    uint8_t src0Gm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                              \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    main_Vec_Unary_test<dtype, OpType::cmdKey>(src0Gm, outputGm, inputParams.stackSize, inputParams.dataSize,      \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow, inputParams.repeatTimes,  \
        inputParams.repeatParams);                                                          \
}

#define REGISTER_VEC_UNARY_SPECIAL(cmdKey, cmdKeyLower, dtype, dtypeKey, dataLength)                                \
TEST_F(VecUnaryTestSuite, VecUnaryTestSuite##_##cmdKey##_##dtype)                                           \
{                                                                                                           \
    uint64_t mask = 256 / sizeof(dtype);                                                                    \
    uint64_t maskHigh = 0xffffffff;                                                                         \
    uint64_t maskLow = 0x000000000;                                                                         \
    uint8_t repeatTimes = dataLength / (256 / sizeof(dtype));                                               \
    InputParams inputParams{dataLength, dataLength, sizeof(dtype), mask, maskHigh, maskLow, repeatTimes,    \
        {1, 1, 8, 8}};                                                                                      \
    uint8_t src0Gm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                              \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    main_Vec_Unary_test<dtype, OpType::cmdKey>(src0Gm, outputGm, inputParams.stackSize, inputParams.dataSize,      \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow, inputParams.repeatTimes,  \
        inputParams.repeatParams);                                                          \
}

// Complete process: Exp -> ExpImpl -> vadd            cmdKey: Exp, cmdKeyLower: add, dtypeKey: s16
// therefore check how many times addimpl has been launched + datacopy result successfully
#define REGISTER_VEC_UNARY_IMPL(cmdKey, cmdKeyLower, dtype, dtypeKey, dataLength)                                      \
    TEST_F(VecUnaryTestSuite, VecUnaryTestSuite_Impl##_##cmdKey##_##dtype)                                             \
    {                                                                                                                  \
        uint64_t mask = 256 / sizeof(dtype);                                                                           \
        uint64_t maskHigh = 0xffffffff;                                                                                \
        uint64_t maskLow = 0x000000000;                                                                                \
        uint8_t repeatTimes = dataLength / (256 / sizeof(dtype));                                                      \
        InputParams inputParams{ dataLength, dataLength, sizeof(dtype), mask,                                          \
            maskHigh,   maskLow,    repeatTimes,   { 1, 1, 8, 8 } };                                                   \
        uint8_t src0Gm[inputParams.stackSize * inputParams.dataTypeSize]{ 0x01 };                                      \
        uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize]{ 0x00 };                                    \
        dtype outputStub[inputParams.stackSize]{ 0x03 };                                                               \
        MOCKER(cmdKey##Impl,                                                                                           \
            void (*)(__ubuf__ dtype *, __ubuf__ dtype *, const uint64_t, const uint8_t, const UnaryRepeatParams &))    \
            .times(1)                                                                                                  \
            .with(outBoundP(outputStub), any(), checkWith(UnaryImplCheckMask<dtype>),                                  \
            checkWith(UnaryImplCheckRepeatTimes<dtype>), checkWith(UnaryImplCheckUnaryRepeatParams));                  \
        MOCKER(cmdKey##Impl,                                                                                           \
            void (*)(__ubuf__ dtype *, __ubuf__ dtype *, const uint64_t[2], const uint8_t, const UnaryRepeatParams &)) \
            .times(1)                                                                                                  \
            .with(outBoundP(outputStub), any(), checkWith(UnaryImplCheckMask2<dtype>),                                 \
            checkWith(UnaryImplCheckRepeatTimes<dtype>), checkWith(UnaryImplCheckUnaryRepeatParams));                  \
        MOCKER(cmdKey##Impl, void (*)(__ubuf__ dtype *, __ubuf__ dtype *, const uint32_t))                             \
            .times(1)                                                                                                  \
            .with(outBoundP(outputStub), any(), checkWith(UnaryImplCheckCalCount));                                    \
        MOCKER(DataCopyUB2GMImpl, void (*)(__gm__ dtype *, __ubuf__ dtype *, const DataCopyParams &, const uint8_t))   \
            .times(1)                                                                                                  \
            .will(invoke(UnaryDataCopyUB2GMImplStub<dtype>));                                                          \
        main_Vec_Unary_test<dtype, OpType::cmdKey>(src0Gm, outputGm, inputParams.stackSize, inputParams.dataSize,                      \
            inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow, inputParams.repeatTimes,         \
            inputParams.repeatParams);                                                                 \
        dtype output[inputParams.stackSize]{ 0x00 };                                                                   \
        dtype *out = output;                                                                                           \
        out = reinterpret_cast<dtype *>(outputGm);                                                                     \
        EXPECT_EQ(out[0], (dtype)0x03);                                                                                \
    }

// dtype unsupported
// Complete process: Exp -> ExpImpl -> return cause invalid dtype, thus triggers [raise] error function
// cmdKey: Exp, cmdKeyLower: add
#define REGISTER_VEC_UNARY_DTYPE_UNSUPPORT(cmdKey, cmdKeyLower, dtype, dataLength)                          \
TEST_F(VecUnaryTestSuite, VecUnaryTestSuite_Unsupport##_##cmdKey##_##dtype)                                 \
{                                                                                                           \
    uint64_t mask = 256 / sizeof(dtype);                                                                    \
    uint64_t maskHigh = 0xffffffff;                                                                         \
    uint64_t maskLow = 0x000000000;                                                                         \
    uint8_t repeatTimes = dataLength / (256 / sizeof(dtype));                                               \
    InputParams inputParams{dataLength, dataLength, sizeof(dtype), mask, maskHigh, maskLow, repeatTimes,    \
        {1, 1, 8, 8}};                                                                                      \
    MOCKER(raise, int(*)(int)).times(3).will(returnValue(0));                                               \
    uint8_t src0Gm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                              \
    uint8_t outputGm[inputParams.stackSize * inputParams.dataTypeSize] { 0x00 };                            \
    main_Vec_Unary_test<dtype, OpType::cmdKey>(src0Gm, outputGm, inputParams.stackSize, inputParams.dataSize,               \
        inputParams.maskCounter, inputParams.maskBitHigh, inputParams.maskBitLow, inputParams.repeatTimes,  \
        inputParams.repeatParams);                                                          \
    EXPECT_EQ(outputGm[0], 0x00);                                                                           \
    EXPECT_EQ(outputGm[1], 0x00);                                                                           \
}

// 决定 args 第几位对应的名字    Ex: SUPPORT_MAP_a  a表示对应的是10个参数的场景
#define PP_ARG_X(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,                                                             \
           a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,                                             \
           A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,XX,...) XX
#define PP_ARG_N(...)                                                                                       \
        PP_ARG_X("ignored", ##__VA_ARGS__,                                                                  \
            Z,Y,X,W,V,U,T,S,R,Q,P,O,N,M,L,K,J,I,H,G,F,E,D,C,B,A,                                            \
            z,y,x,w,v,u,t,s,r,q,p,o,n,m,l,k,j,i,h,g,f,e,d,c,b,a,                                            \
                                            9,8,7,6,5,4,3,2,1,0)

// 对于support的dtype， 校验 vxxx ut + xxxImpl ut用例
#define Support(cmdKey, cmdKeyLower, dtype, str)                   \
    REGISTER_VEC_UNARY(cmdKey, cmdKeyLower, dtype, str, 256);      \
    REGISTER_VEC_UNARY_IMPL(cmdKey, cmdKeyLower, dtype, str, 256); \
// Rsqrt/Rec deleted, use 1/sqrt, 1/x instead
#define SupportSpecial(cmdKey, cmdKeyLower, dtype, str)               \
    REGISTER_VEC_UNARY_SPECIAL(cmdKey, cmdKeyLower, dtype, str, 256); \
    REGISTER_VEC_UNARY_IMPL(cmdKey, cmdKeyLower, dtype, str, 256);
// 对于unsupport的dtype, 校验是否raise error ut用例
#define Unsupport(cmdKey, cmdKeyLower, dtype)                                                               \
    REGISTER_VEC_UNARY_DTYPE_UNSUPPORT(cmdKey, cmdKeyLower, dtype, 256);                                    \

#define PP_VA_NAME(prefix,...) \
        PP_CAT2(prefix,PP_ARG_N(__VA_ARGS__))
#define PP_CAT2(a,b)      PP_CAT2_1(a,b)
#define PP_CAT2_1(a,b)    a##b

// 因为Support的UT除去指令类型cmdkey + cmdKeyLower Ex: Exp + exp，还需要 dtype + shortDtype Ex： float + f32
// 因此需要的参数都是除去cmdkey + cmdKeyLower，都是2个一组的取dtype相关的参数
// 假设Add支持的数据类型是 half + float + int32_t，那么对应的宏的展开逻辑是
// SUPPORT_MAP(Add, add, half, f16, float, f32, int32_t, s32)   等价于SUPPORT_MAP_8(Add, add, half, f16, float, f32, int32_t, s32) 因为一共8个参数
// => Support(Add, add, half, f16)                         取出最开始的2个参数，然后展开Support宏
//  + SUPPORT_MAP_6(Add, add, float, f32, int32_t, s32)    float, f32, int32_t, s32作为__VA_ARGS__继续向下传，此时6个参数
// => Support(Add, add, float, f32)                        取出此时最开始的2个参数，然后展开Support宏
//  + SUPPORT_MAP_4(Add, add, int32_t, s32)                int32_t, s32作为__VA_ARGS__继续向下传，此时4个参数，以此类推
#define SUPPORT_MAP_4(cmdKey, cmdKeyLower, dtype, shortDtype)      Support(cmdKey, cmdKeyLower, dtype, shortDtype)
#define SUPPORT_MAP_6(cmdKey, cmdKeyLower, dtype, shortDtype, ...) Support(cmdKey, cmdKeyLower, dtype, shortDtype) SUPPORT_MAP_4(cmdKey, cmdKeyLower, __VA_ARGS__)
#define SUPPORT_MAP_8(cmdKey, cmdKeyLower, dtype, shortDtype, ...) Support(cmdKey, cmdKeyLower, dtype, shortDtype) SUPPORT_MAP_6(cmdKey, cmdKeyLower, __VA_ARGS__)
#define SUPPORT_MAP_a(cmdKey, cmdKeyLower, dtype, shortDtype, ...) Support(cmdKey, cmdKeyLower, dtype, shortDtype) SUPPORT_MAP_8(cmdKey, cmdKeyLower, __VA_ARGS__)
#define SUPPORT_MAP_c(cmdKey, cmdKeyLower, dtype, shortDtype, ...) Support(cmdKey, cmdKeyLower, dtype, shortDtype) SUPPORT_MAP_a(cmdKey, cmdKeyLower, __VA_ARGS__)
#define SUPPORT_MAP_e(cmdKey, cmdKeyLower, dtype, shortDtype, ...) Support(cmdKey, cmdKeyLower, dtype, shortDtype) SUPPORT_MAP_c(cmdKey, cmdKeyLower, __VA_ARGS__)
#define SUPPORT_MAP_g(cmdKey, cmdKeyLower, dtype, shortDtype, ...) Support(cmdKey, cmdKeyLower, dtype, shortDtype) SUPPORT_MAP_e(cmdKey, cmdKeyLower, __VA_ARGS__)
#define SUPPORT_MAP_i(cmdKey, cmdKeyLower, dtype, shortDtype, ...) Support(cmdKey, cmdKeyLower, dtype, shortDtype) SUPPORT_MAP_g(cmdKey, cmdKeyLower, __VA_ARGS__)

// 因为Unsupport的UT除去指令类型cmdkey + cmdKeyLower Ex: Exp + exp，只需要一个dtype Ex: float
// 因此需要的参数都是除去cmdkey + cmdKeyLower，都是1个一组的取dtype相关的参数
// 假设Add不支持的数据类型是 half + float + int32_t，那么对应的宏的展开逻辑是
// UNSUPPORT_MAP(Add, add, half, float, int32_t)        等价于UNSUPPORT_MAP_5(Add, add, half, float, int32_t) 因为一共5个参数
// => Unsupport(Add, add, half)                         取出最开始的1个参数，然后展开Unsupport宏
//  + SUPPORT_MAP_4(Add, add, float, int32_t)           float, int32_t作为__VA_ARGS__继续向下传，此时4个参数
// => Unsupport(Add, add, float)                        取出此时最开始的1个参数，然后展开Support宏
//  + SUPPORT_MAP_3(Add, add, int32_t)                  int32_t作为__VA_ARGS__继续向下传，此时3个参数，以此类推
#define UNSUPPORT_MAP_3(cmdKey, cmdKeyLower, dtype)      Unsupport(cmdKey, cmdKeyLower, dtype)
#define UNSUPPORT_MAP_4(cmdKey, cmdKeyLower, dtype, ...) Unsupport(cmdKey, cmdKeyLower, dtype) UNSUPPORT_MAP_3(cmdKey, cmdKeyLower, __VA_ARGS__)
#define UNSUPPORT_MAP_5(cmdKey, cmdKeyLower, dtype, ...) Unsupport(cmdKey, cmdKeyLower, dtype) UNSUPPORT_MAP_4(cmdKey, cmdKeyLower, __VA_ARGS__)
#define UNSUPPORT_MAP_6(cmdKey, cmdKeyLower, dtype, ...) Unsupport(cmdKey, cmdKeyLower, dtype) UNSUPPORT_MAP_5(cmdKey, cmdKeyLower, __VA_ARGS__)
#define UNSUPPORT_MAP_7(cmdKey, cmdKeyLower, dtype, ...) Unsupport(cmdKey, cmdKeyLower, dtype) UNSUPPORT_MAP_6(cmdKey, cmdKeyLower, __VA_ARGS__)
#define UNSUPPORT_MAP_8(cmdKey, cmdKeyLower, dtype, ...) Unsupport(cmdKey, cmdKeyLower, dtype) UNSUPPORT_MAP_7(cmdKey, cmdKeyLower, __VA_ARGS__)
#define UNSUPPORT_MAP_9(cmdKey, cmdKeyLower, dtype, ...) Unsupport(cmdKey, cmdKeyLower, dtype) UNSUPPORT_MAP_8(cmdKey, cmdKeyLower, __VA_ARGS__)
#define UNSUPPORT_MAP_a(cmdKey, cmdKeyLower, dtype, ...) Unsupport(cmdKey, cmdKeyLower, dtype) UNSUPPORT_MAP_9(cmdKey, cmdKeyLower, __VA_ARGS__)
#define UNSUPPORT_MAP_b(cmdKey, cmdKeyLower, dtype, ...) Unsupport(cmdKey, cmdKeyLower, dtype) UNSUPPORT_MAP_a(cmdKey, cmdKeyLower, __VA_ARGS__)

// SUPPORT_MAP(x, y, z) 等价于调用 SUPPORT_MAP_3(x, y, z)
#define SUPPORT_MAP(...)     PP_VA_NAME(SUPPORT_MAP_, __VA_ARGS__)(__VA_ARGS__)
#define UNSUPPORT_MAP(...)   PP_VA_NAME(UNSUPPORT_MAP_, __VA_ARGS__)(__VA_ARGS__)

// Rsqrt/Rec deleted, use 1/sqrt, 1/x instead
SUPPORT_MAP(Abs, abs, half, f16, float, f32, int16_t, s16, int32_t, s32);
SUPPORT_MAP(Not, not, uint16_t, u16, int16_t, s16, half, f16, float, f32, uint32_t, u32, int32_t, s32);
SUPPORT_MAP(Relu, relu, half, f16, float, f32, int32_t, s32);