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
template <typename T> class BilinearInterpolationTest {
public:
    __aicore__ inline BilinearInterpolationTest() {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR src0Gm, GM_ADDR src0OffsetGm, GM_ADDR src1Gm, uint64_t mask,
        uint8_t hRepeat, bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat)
    {
        mMask = mask;
        mHRepeat = hRepeat;
        mRepeatMode = repeatMode;
        mDstBlkStride = dstBlkStride;
        mVROffset = vROffset;
        mVRepeat = vRepeat;

        mDstElementCount = vRepeat * vROffset;
        mSrc0ElementCount = hRepeat * vRepeat * 128;
        mSrc0OffsetlementCount = hRepeat * vRepeat * 8; // 每个repeat需要offset8个数进行gatherb
        // 向上对齐到16
        mSrc0OffsetlementCount =
            mSrc0OffsetlementCount % 16 == 0 ? mSrc0OffsetlementCount : (mSrc0OffsetlementCount / 16 + 1) * 16;
        if (mRepeatMode) {
            uint32_t src1Number = hRepeat * vRepeat * 8;
            mSrc1lementCount = src1Number % 16 == 0 ? src1Number : (src1Number / 16 + 1) * 16;
        } else {
            uint32_t src1Number = hRepeat * vRepeat;
            mSrc1lementCount = src1Number % 16 == 0 ? src1Number : (src1Number / 16 + 1) * 16;
        }

        mDstGlobal.SetGlobalBuffer((__gm__ T *)dstGm);
        mSrc0Global.SetGlobalBuffer((__gm__ T *)src0Gm);
        mSrc0OffsetGlobal.SetGlobalBuffer((__gm__ uint32_t *)src0OffsetGm);
        mSrc1Global.SetGlobalBuffer((__gm__ T *)src1Gm);

        mPipe.InitBuffer(mQueOut, 1, mDstElementCount * sizeof(T));
        mPipe.InitBuffer(mQueInSrc0, 1, mSrc0ElementCount * sizeof(T));
        mPipe.InitBuffer(mQueInSrc0Offset, 1, mSrc0OffsetlementCount * sizeof(uint32_t));
        mPipe.InitBuffer(mQueInSrc1, 1, mSrc1lementCount * sizeof(T));
        mPipe.InitBuffer(mQueTmp, 1, 81920);
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
        LocalTensor<T> src0Local = mQueInSrc0.AllocTensor<T>();
        DataCopy(src0Local, mSrc0Global, mSrc0ElementCount);
        mQueInSrc0.EnQue(src0Local);

        LocalTensor<uint32_t> src0OffsetLocal = mQueInSrc0Offset.AllocTensor<uint32_t>();
        DataCopy(src0OffsetLocal, mSrc0OffsetGlobal, mSrc0OffsetlementCount);
        mQueInSrc0Offset.EnQue(src0OffsetLocal);

        LocalTensor<T> src1Local = mQueInSrc1.AllocTensor<T>();
        DataCopy(src1Local, mSrc1Global, mSrc1lementCount);
        mQueInSrc1.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> dstLocal = mQueOut.AllocTensor<T>();
        LocalTensor<uint8_t> tmpLocal = mQueTmp.AllocTensor<uint8_t>();
        LocalTensor<T> src0Local = mQueInSrc0.DeQue<T>();
        LocalTensor<uint32_t> src0OffsetLocal = mQueInSrc0Offset.DeQue<uint32_t>();
        LocalTensor<T> src1Local = mQueInSrc1.DeQue<T>();

        Duplicate(dstLocal, static_cast<T>(0), mDstElementCount);
        uint64_t maskbit[2] = {0x0, 0xFFFFFFFFFFFFFFFF};
        BilinearInterpolation(dstLocal, src0Local, src0OffsetLocal, src1Local, maskbit, mHRepeat, mRepeatMode,
            mDstBlkStride, mVROffset, mVRepeat, tmpLocal);
        BilinearInterpolation(dstLocal, src0Local, src0OffsetLocal, src1Local, mMask, mHRepeat, mRepeatMode,
            mDstBlkStride, mVROffset, mVRepeat, tmpLocal);
        mQueOut.EnQue(dstLocal);
        mQueTmp.FreeTensor(tmpLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = mQueOut.DeQue<T>();
        DataCopy(mDstGlobal, dstLocal, mDstElementCount);
        mQueOut.FreeTensor(dstLocal);
    }

private:
    uint32_t mDstElementCount;
    uint32_t mSrc0ElementCount;
    uint32_t mSrc0OffsetlementCount;
    uint32_t mSrc1lementCount;
    uint64_t mMask;
    uint8_t mHRepeat;
    bool mRepeatMode;
    uint16_t mDstBlkStride;
    uint16_t mVROffset;
    uint8_t mVRepeat;

    TPipe mPipe;
    GlobalTensor<T> mDstGlobal;
    GlobalTensor<T> mSrc0Global;
    GlobalTensor<uint32_t> mSrc0OffsetGlobal;
    GlobalTensor<T> mSrc1Global;

    TQue<TPosition::VECOUT, 1> mQueOut;
    TQue<TPosition::VECIN, 1> mQueInSrc0;
    TQue<TPosition::VECIN, 1> mQueInSrc0Offset;
    TQue<TPosition::VECIN, 1> mQueInSrc1;
    TQue<TPosition::VECIN, 1> mQueTmp;
}; // class BilinearInterpolationTest
} // namespace AscendC

template <typename T>
__global__ __aicore__ void testBilinearInterpolation(GM_ADDR dstGm, GM_ADDR src0Gm, GM_ADDR src0OffsetGm,
    GM_ADDR src1Gm, uint64_t mask, uint8_t hRepeat, bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset,
    uint8_t vRepeat)
{
    AscendC::BilinearInterpolationTest<T> op;
    op.Init(dstGm, src0Gm, src0OffsetGm, src1Gm, mask, hRepeat, repeatMode, dstBlkStride, vROffset, vRepeat);
    op.Process();
}

struct BilinearInterpolationParams {
    uint64_t mask;
    uint8_t hRepeat;
    bool repeatMode;
    uint16_t dstBlkStride;
    uint16_t vROffset;
    uint8_t vRepeat;
    void (*cal_func)(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint64_t, uint8_t, bool, uint16_t, uint16_t, uint8_t);
};

class BilinearInterpolationTestsuite : public testing::Test,
    public testing::WithParamInterface<BilinearInterpolationParams> {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(2);
    }
    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_BILINEARINTERPOLATION, BilinearInterpolationTestsuite,
    ::testing::Values(BilinearInterpolationParams{ 128, 2, false, 1, 128, 2, testBilinearInterpolation<half> }));

TEST_P(BilinearInterpolationTestsuite, testBilinearInterpolation)
{
    auto param = GetParam();
    auto vRepeat = param.vRepeat;
    auto hRepeat = param.hRepeat;
    auto vROffset = param.vROffset;
    uint32_t dstElementCount = vRepeat * vROffset;
    uint32_t src0ElementCount = hRepeat * vRepeat * 128;
    uint32_t src0OffsetlementCount = hRepeat * vRepeat * 8; // 每个repeat需要offset8个数进行gatherb
    // 向上对齐到16
    src0OffsetlementCount =
        src0OffsetlementCount % 16 == 0 ? src0OffsetlementCount : (src0OffsetlementCount / 16 + 1) * 16;
    uint32_t src1lementCount;
    if (param.repeatMode) {
        uint32_t src1Number = hRepeat * vRepeat * 8;
        src1lementCount = src1Number % 16 == 0 ? src1Number : (src1Number / 16 + 1) * 16;
    } else {
        uint32_t src1Number = hRepeat * vRepeat;
        src1lementCount = src1Number % 16 == 0 ? src1Number : (src1Number / 16 + 1) * 16;
    }
    uint8_t src0Gm[src0ElementCount * sizeof(half)] = {0};
    uint8_t src0OffsetGm[src0OffsetlementCount * sizeof(uint32_t)] = {0};
    uint8_t src1Gm[src1lementCount * sizeof(half)] = {0};
    uint8_t dstGm[dstElementCount * sizeof(half)] = {0};
    param.cal_func(dstGm, src0Gm, src0OffsetGm, src1Gm, param.mask, param.hRepeat, param.repeatMode, param.dstBlkStride,
        param.vROffset, param.vRepeat);

    for (int32_t i = 0; i < dstElementCount; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}