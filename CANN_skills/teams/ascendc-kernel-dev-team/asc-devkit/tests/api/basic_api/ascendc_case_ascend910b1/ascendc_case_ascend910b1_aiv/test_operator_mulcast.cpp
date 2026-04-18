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

namespace AscendC {
template <typename T, typename U>
class MulcastTest {
public:
    __aicore__ inline MulcastTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* src0Gm,
        __gm__ uint8_t* src1Gm, const uint32_t calCount)
    {
        mElementCount = calCount;
        mSrc0Global.SetGlobalBuffer((__gm__ U*)src0Gm);
        mSrc1Global.SetGlobalBuffer((__gm__ U*)src1Gm);
        mDstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);

        mPipe.InitBuffer(mQueInSrc0, 1, mElementCount * sizeof(U));
        mPipe.InitBuffer(mQueInSrc1, 1, mElementCount * sizeof(U));
        mPipe.InitBuffer(mQueOut, 1, mElementCount * sizeof(T));
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
        LocalTensor<U> src0Local = mQueInSrc0.AllocTensor<U>();
        DataCopy(src0Local, mSrc0Global, mElementCount);
        mQueInSrc0.EnQue(src0Local);

        LocalTensor<U> src1Local = mQueInSrc1.AllocTensor<U>();
        DataCopy(src1Local, mSrc1Global, mElementCount);
        mQueInSrc1.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<U> src0Local = mQueInSrc0.DeQue<U>();
        LocalTensor<U> src1Local = mQueInSrc1.DeQue<U>();
        LocalTensor<T> dstLocal = mQueOut.AllocTensor<T>();

        BinaryRepeatParams repeatParams;
        repeatParams.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
        MulCast(dstLocal, src0Local, src1Local, 128, mElementCount / 128, repeatParams);
        uint64_t maskbit[2] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
        MulCast(dstLocal, src0Local, src1Local, maskbit, mElementCount / 128, repeatParams);
        MulCast(dstLocal, src0Local, src1Local, mElementCount);
        mQueInSrc0.FreeTensor(src0Local);
        mQueInSrc1.FreeTensor(src1Local);
        mQueOut.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = mQueOut.DeQue<T>();
        DataCopy(mDstGlobal, dstLocal, mElementCount);
        mQueOut.FreeTensor(dstLocal);
    }
private:
    TPipe mPipe;
    uint32_t mElementCount;
    GlobalTensor<U> mSrc0Global;
    GlobalTensor<U> mSrc1Global;
    GlobalTensor<T> mDstGlobal;
    TQue<TPosition::VECIN, 1> mQueInSrc0;
    TQue<TPosition::VECIN, 1> mQueInSrc1;
    TQue<TPosition::VECOUT, 1> mQueOut;
}; // class MulcastTest
} // namespace AscendC

template <typename T, typename U>
__global__ __aicore__ void testMulcast(GM_ADDR dstGm, GM_ADDR src0Gm, GM_ADDR src1Gm, uint32_t calCount)
{
    AscendC::MulcastTest<T, U> op;
    op.Init(dstGm, src0Gm, src1Gm, calCount);
    op.Process();
}

struct mulcastParams {
    uint32_t calCount;
    bool expectRes;
    uint8_t srcTypeSize;
    uint8_t dstTypeSize;
    void (*CalFunc)(uint8_t*, uint8_t*, uint8_t*, uint32_t);
};

class MulcastTestsuite : public testing::Test,
    public testing::WithParamInterface<mulcastParams> {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
        GlobalMockObject::verify();
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_MULCAST, MulcastTestsuite,
    ::testing::Values(mulcastParams { 128, true, 2, 1, testMulcast<int8_t, half>},
        // src必须为half, dst必须为int8_t / uint8_t
        mulcastParams { 128, false, 2, 2, testMulcast<uint16_t, half>},
        mulcastParams { 128, true, 2, 1, testMulcast<uint8_t, half>},
        mulcastParams { 128, false, 2, 1, testMulcast<int8_t, uint16_t>}
));

TEST_P(MulcastTestsuite, testMulcast)
{
    auto param = GetParam();
    auto srcTypeSize = param.srcTypeSize;
    auto dstTypeSize = param.dstTypeSize;
    uint8_t src0Gm[param.calCount * srcTypeSize] = {0};
    uint8_t src1Gm[param.calCount * srcTypeSize] = {0};
    uint8_t dstGm[param.calCount * dstTypeSize] = {0};
    if (!param.expectRes) {
        MOCKER(raise, int(*)(int)).times(3).will(returnValue(0));
    }
    param.CalFunc(dstGm, src0Gm, src1Gm, param.calCount);
    for (int32_t i = 0; i < param.calCount; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}