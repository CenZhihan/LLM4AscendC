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

namespace AscendC {
template <typename T, int mode>
class MicroReduceMaxTest {
public:
    __aicore__ inline MicroReduceMaxTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
        __gm__ uint8_t* indexOffsetGm, const uint32_t count)
    {
        mElementCount = count;
        mSrcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        mIndexOffsetGlobal.SetGlobalBuffer((__gm__ T*)indexOffsetGm);
        mDstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
 
        mPipe.InitBuffer(mQueIn, 1, mElementCount * sizeof(T));
        mPipe.InitBuffer(mQueIn2, 1, mElementCount * sizeof(T));
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
        LocalTensor<T> srcLocal = mQueIn.AllocTensor<T>();
        DataCopy(srcLocal, mSrcGlobal, mElementCount);
        mQueIn.EnQue(srcLocal);
 
        LocalTensor<T> indexLocal = mQueIn2.AllocTensor<T>();
        DataCopy(indexLocal, mIndexOffsetGlobal, mElementCount);
        mQueIn2.EnQue(indexLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = mQueIn.DeQue<T>();
        LocalTensor<T> indexLocal = mQueIn2.DeQue<T>();
        LocalTensor<T> dstLocal = mQueOut.AllocTensor<T>();
        __ubuf__ T* srcPtr = (__ubuf__ T*)srcLocal.GetPhyAddr();
        __ubuf__ T* indexPtr = (__ubuf__ T*)indexLocal.GetPhyAddr();
        __ubuf__ T* dstPtr = (__ubuf__ T*)dstLocal.GetPhyAddr();
 
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> vIndexReg;
            Reg::RegTensor<T> vDstReg;
            Reg::RegTensor<T> vSrcReg;
            uint32_t sreg = static_cast<uint32_t>(mElementCount);
            Reg::MaskReg preg;
            uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(T);
            uint16_t repeatTimes = CeilDivision(mElementCount, repeatElm);
            T scalar = 1;
            for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
                preg = Reg::UpdateMask<half>(sreg);
                Reg::DataCopy(vIndexReg, indexPtr + i * repeatElm);
                Reg::ReduceMax(vDstReg, vSrcReg, preg);
                Reg::DataCopy(dstPtr + i * repeatElm, vDstReg, preg);
            }
        }
       
 
        mQueIn.FreeTensor(srcLocal);
        mQueIn2.FreeTensor(indexLocal);
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
    GlobalTensor<T> mSrcGlobal;
    GlobalTensor<T> mIndexOffsetGlobal;
    GlobalTensor<T> mDstGlobal;
    TQue<TPosition::VECIN, 1> mQueIn;
    TQue<TPosition::VECIN, 1> mQueIn2;
    TQue<TPosition::VECOUT, 1> mQueOut;
};
} // namespace AscendC

template <typename T, int mode>
__global__ __aicore__ void testReduceMax(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
    __gm__ uint8_t* indexOffsetGm, uint32_t elementCount)
{
    AscendC::MicroReduceMaxTest<T, mode> op;
    op.Init(dstGm, srcGm, indexOffsetGm, elementCount);
    op.Process();
}

struct ReduceMaxParams {
    uint32_t dstTypeSize;
    uint32_t srcTypeSize;
    uint32_t elementCount;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint32_t);
};

class reduceMaxTestsuite : public testing::Test,
    public testing::WithParamInterface<ReduceMaxParams> {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_REDUCE_MAX, reduceMaxTestsuite,
    ::testing::Values(ReduceMaxParams { 2, 2, 128, testReduceMax<half, 0>},
    ReduceMaxParams { 4, 4, 64, testReduceMax<int32_t, 0>},
    ReduceMaxParams { 4, 4, 64, testReduceMax<uint32_t, 0>},
    ReduceMaxParams { 4, 4, 64, testReduceMax<int16_t, 0>},
    ReduceMaxParams { 4, 4, 64, testReduceMax<uint16_t, 0>},
    ReduceMaxParams { 4, 4, 64, testReduceMax<float, 0>}));


TEST_P(reduceMaxTestsuite, testReduceMax)
{
    auto param = GetParam();
    uint8_t dstGm[param.elementCount * param.dstTypeSize] = {0};
    uint8_t srcGm[param.elementCount * param.srcTypeSize] = {0};
    uint8_t indexOffsetGm[param.elementCount * param.dstTypeSize] = {0};
    param.cal_func(dstGm, srcGm, indexOffsetGm, param.elementCount);
    
    for (int32_t i = 0; i < param.elementCount; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}