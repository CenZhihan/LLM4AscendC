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
#include "kernel_utils.h"

#include "kernel_operator.h"

namespace AscendC {
template <typename T>
class CreateVecIndexTest {
public:
    __aicore__ inline CreateVecIndexTest() {}
    __aicore__ inline void Init(GM_ADDR dstGm, uint64_t mask, uint8_t repeatTimes,
        uint16_t dstBlkStride, uint8_t dstRepStride)
    {
        mMask = mask;
        mRepeatTimes = repeatTimes;
        mDstBlkStride = dstBlkStride;
        mDstRepStride = dstRepStride;
        mElementCount = mDstBlkStride * mDstRepStride * 32 * mRepeatTimes / sizeof(T);
        mDstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        mPipe.InitBuffer(mQueOut, 1, mDstBlkStride * mDstRepStride * 32 * mRepeatTimes);
        mPipe.InitBuffer(mQueTmp, 1, 1024);
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
        ;
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> dstLocal = mQueOut.AllocTensor<T>();
        LocalTensor<uint8_t> tmpLocal = mQueTmp.AllocTensor<uint8_t>();
        Duplicate(dstLocal, static_cast<T>(0), mElementCount);
        pipe_barrier(PIPE_ALL);
        uint64_t maskbit[2] = {0xFFFFFFFFFFFFFFFF, 0};
        CreateVecIndex(dstLocal, static_cast<T>(0), maskbit, mRepeatTimes, mDstBlkStride, mDstRepStride);
        CreateVecIndex(dstLocal, static_cast<T>(0), mMask, mRepeatTimes, mDstBlkStride, mDstRepStride);
        CreateVecIndex(dstLocal, static_cast<T>(0), mRepeatTimes * 256 / sizeof(T));
        mQueOut.EnQue(dstLocal);
        mQueTmp.FreeTensor(tmpLocal);
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
    uint32_t mMask;
    uint32_t mRepeatTimes;
    uint32_t mDstBlkStride;
    uint32_t mDstRepStride;
    GlobalTensor<T> mDstGlobal;
    TQue<TPosition::VECOUT, 1> mQueOut;
    TQue<TPosition::VECIN, 1> mQueTmp;
}; // class CreateVecIndexTest
} // namespace AscendC

template <typename T>
__global__ __aicore__ void testCreateVecIndex(GM_ADDR dstGm, uint64_t mask, uint8_t repeatTimes,
        uint16_t dstBlkStride, uint8_t dstRepStride)
{
    AscendC::CreateVecIndexTest<T> op;
    op.Init(dstGm, mask, repeatTimes, dstBlkStride, dstRepStride);
    op.Process();
}

struct CreateVecIndexParams {
    uint32_t typeSize;
    uint64_t mask;
    uint8_t repeatTimes;
    uint16_t dstBlkStride;
    uint8_t dstRepStride;
    void (*cal_func)(uint8_t*, uint64_t, uint8_t, uint16_t, uint8_t);
};

class CreateVecIndexTestsuite : public testing::Test,
    public testing::WithParamInterface<CreateVecIndexParams> {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_CREATEVECINDEX, CreateVecIndexTestsuite,
    ::testing::Values(CreateVecIndexParams { 2, 256, 2, 1, 8, testCreateVecIndex<half>},
                      CreateVecIndexParams { 4, 128, 2, 1, 8, testCreateVecIndex<float>},
                      CreateVecIndexParams { 2, 256, 2, 1, 8, testCreateVecIndex<int16_t>},
                      CreateVecIndexParams { 4, 128, 2, 1, 8, testCreateVecIndex<int32_t>}));

TEST_P(CreateVecIndexTestsuite, testCreateVecIndex)
{
    auto param = GetParam();
    uint8_t dstGm[param.repeatTimes * 256] = {0};
    param.cal_func(dstGm, param.mask, param.repeatTimes, param.dstBlkStride, param.dstRepStride);
    
    for (int32_t i = 0; i < param.repeatTimes * 256 / param.typeSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}