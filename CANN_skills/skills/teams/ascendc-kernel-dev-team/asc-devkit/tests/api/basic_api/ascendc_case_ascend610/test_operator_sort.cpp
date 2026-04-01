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
// #include "model/model_factory.h"
#include "common.h"

namespace AscendC {
template <typename T, bool isFullSort, bool isExhaustedSuspension>
class FullSort {
public:
    __aicore__ inline FullSort() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstValueGm, __gm__ uint8_t* dstIndexGm,
        __gm__ uint8_t* srcValueGm, __gm__ uint8_t* srcIndexGm,
        uint32_t elementCount)
    {
        mElementCount = elementCount;
        mConcatRepeatTimes = mElementCount / 16;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        mSortRepeatTimes = mElementCount / 32;
        mExtractRepeatTimes = mElementCount / 32;
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002)
        mSortRepeatTimes = mElementCount / 16;
        mExtractRepeatTimes = mElementCount / 16;
#endif
        mValueGlobal.SetGlobalBuffer((__gm__ T*)srcValueGm);
        mIndexGlobal.SetGlobalBuffer((__gm__ uint32_t*)srcIndexGm);
        mDstValueGlobal.SetGlobalBuffer((__gm__ T*)dstValueGm);
        mDstIndexGlobal.SetGlobalBuffer((__gm__ uint32_t*)dstIndexGm);
        mPipe.InitBuffer(mQueIn, 2, mElementCount * sizeof(uint32_t));
        mPipe.InitBuffer(mQueOut, 2, mElementCount * sizeof(uint32_t));
        uint32_t concatTmpLocalSize = mElementCount * 8 * sizeof(T);
        mPipe.InitBuffer(mQueCalc, 1, mElementCount * 8 * sizeof(T));
        mPipe.InitBuffer(mQueTmp, 2, concatTmpLocalSize);
        mPipe.InitBuffer(tbuf, mElementCount * 8 * sizeof(T)); // 1 proposal = 8 elements
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
        LocalTensor<T> valueLocal = mQueIn.AllocTensor<T>();
        DataCopy(valueLocal, mValueGlobal, mElementCount);
        mQueIn.EnQue(valueLocal);

        LocalTensor<uint32_t> indexLocal = mQueIn.AllocTensor<uint32_t>();
        DataCopy(indexLocal, mIndexGlobal, mElementCount);
        mQueIn.EnQue(indexLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> valueLocal = mQueIn.DeQue<T>();
        LocalTensor<uint32_t> indexLocal = mQueIn.DeQue<uint32_t>();
        LocalTensor<T> sortedLocal = mQueCalc.AllocTensor<T>();
        LocalTensor<T> concatTmpLocal = mQueTmp.AllocTensor<T>();
        LocalTensor<T> sortTmpLocal = mQueTmp.AllocTensor<T>();
        LocalTensor<T> dstValueLocal = mQueOut.AllocTensor<T>();
        LocalTensor<uint32_t> dstIndexLocal = mQueOut.AllocTensor<uint32_t>();
        LocalTensor<T> concatLocal = tbuf.Get<T>();

        valueLocal.SetSize(mElementCount);
        Concat(concatLocal, valueLocal, concatTmpLocal, mConcatRepeatTimes);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        if (sizeof(T) == sizeof(half)) {
            sortedLocal.SetSize(mElementCount * 4);
            sortTmpLocal.SetSize(mElementCount * 4);
        } else {
            sortedLocal.SetSize(mElementCount * 2);
            sortTmpLocal.SetSize(mElementCount * 2);
        }
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002)
        sortedLocal.SetSize(mElementCount * 8);
        sortTmpLocal.SetSize(mElementCount * 8);
#endif
        Sort<T, isFullSort>(sortedLocal, concatLocal, indexLocal, sortTmpLocal, mSortRepeatTimes);
        
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        uint32_t singleMergeTmpElementCount = 32;
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002)
        uint32_t singleMergeTmpElementCount = 16;
#endif
        uint32_t baseOffset;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        baseOffset = singleMergeTmpElementCount * 8 / sizeof(T);
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002)
        if constexpr (sizeof(T) == sizeof(half)) {
            baseOffset = singleMergeTmpElementCount * 16 / sizeof(T);
        } else {
            baseOffset = singleMergeTmpElementCount * 32 / sizeof(T);
        }
#endif
        MrgSortSrcList sortList = MrgSortSrcList(sortedLocal[0], sortedLocal[baseOffset], sortedLocal[2 * baseOffset], sortedLocal[3 * baseOffset]);
        const uint16_t elementCountList[4] = {singleMergeTmpElementCount, singleMergeTmpElementCount, singleMergeTmpElementCount, singleMergeTmpElementCount};
        uint32_t sortedNum[4];
        MrgSort<T, isExhaustedSuspension>(sortTmpLocal, sortList, elementCountList, sortedNum, 0b1111, 1);

        uint16_t mrgRes1 = 0;
        uint16_t mrgRes2 = 0;
        uint16_t mrgRes3 = 0;
        uint16_t mrgRes4 = 0;
        GetMrgSortResult(mrgRes1, mrgRes2, mrgRes3, mrgRes4);

        mQueTmp.FreeTensor(concatTmpLocal);
        mQueTmp.FreeTensor(sortTmpLocal);
        mQueIn.FreeTensor(valueLocal);
        mQueIn.FreeTensor(indexLocal);
        Extract(dstValueLocal, dstIndexLocal, sortedLocal, mExtractRepeatTimes);
        mQueOut.EnQue(dstValueLocal);
        mQueOut.EnQue(dstIndexLocal);
        
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstValueLocal = mQueOut.DeQue<T>();
        LocalTensor<uint32_t> dstIndexLocal = mQueOut.DeQue<uint32_t>();
        DataCopy(mDstValueGlobal, dstValueLocal, mElementCount);
        DataCopy(mDstIndexGlobal, dstIndexLocal, mElementCount);
        mQueOut.FreeTensor(dstValueLocal);
        mQueOut.FreeTensor(dstIndexLocal);
    }
private:
    TPipe mPipe;
    TQue<TPosition::VECIN, 2> mQueIn;
    TQue<TPosition::VECOUT, 2> mQueOut;
    TQue<TPosition::VECIN, 1> mQueTmp;
    TQue<TPosition::VECIN, 1> mQueCalc;
    TBuf<TPosition::VECCALC> tbuf;
    GlobalTensor<T> mValueGlobal;
    GlobalTensor<uint32_t> mIndexGlobal;
    GlobalTensor<T> mDstValueGlobal;
    GlobalTensor<uint32_t> mDstIndexGlobal;
    uint32_t mElementCount;
    uint32_t mConcatRepeatTimes;
    uint32_t mSortRepeatTimes;
    uint32_t mExtractRepeatTimes;
}; // class FullSort
} // namespace AscendC

template <typename T, bool isFullSort, bool isExhaustedSuspension>
__global__ __aicore__ void testSort(__gm__ uint8_t* dstValueGm, __gm__ uint8_t* dstIndexGm, __gm__ uint8_t* srcValueGm, __gm__ uint8_t* srcIndexGm,
    uint32_t elementCount)
{
    AscendC::FullSort<T, isFullSort, isExhaustedSuspension> op;
    op.Init(dstValueGm, dstIndexGm, srcValueGm, srcIndexGm, elementCount);
    op.Process();
}

struct SortParams {
    uint32_t typeSize;
    uint32_t elementCount;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint32_t);
};

class SortTestsuite : public testing::Test,
    public testing::WithParamInterface<SortParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
    INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_SORT, SortTestsuite,
        ::testing::Values(SortParams { 2, 128, testSort<half, true, true>},
            SortParams { 2, 128, testSort<half, true, false>},
            SortParams { 4, 128, testSort<float, true, true>},
            SortParams { 4, 128, testSort<float, true, false>},
            SortParams { 2, 192, testSort<half, true, true>},
            SortParams { 2, 192, testSort<half, true, false>},
            SortParams { 4, 512, testSort<float, true, true>},
            SortParams { 4, 512, testSort<float, true, false>},
            SortParams { 2, 512, testSort<half, true, true>},
            SortParams { 2, 512, testSort<half, true, false>},
            SortParams { 4, 192, testSort<float, true, true>},
            SortParams { 4, 192, testSort<float, true, false>},
            SortParams { 2, 768, testSort<half, true, true>},
            SortParams { 2, 768, testSort<half, true, false>},
            SortParams { 4, 768, testSort<float, true, true>},
            SortParams { 4, 768, testSort<float, true, false>}));
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 1001)
    INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_SORT, SortTestsuite,
        ::testing::Values(SortParams { 2, 128, testSort<half, true, true>},
            SortParams { 2, 128, testSort<half, true, false>},
            SortParams { 2, 192, testSort<half, true, true>},
            SortParams { 2, 192, testSort<half, true, false>},
            SortParams { 2, 512, testSort<half, true, true>},
            SortParams { 2, 512, testSort<half, true, false>},
            SortParams { 2, 768, testSort<half, true, true>},
            SortParams { 2, 768, testSort<half, true, false>} ));
#endif

TEST_P(SortTestsuite, testSort)
{
    auto param = GetParam();
    uint8_t srcValueGm[param.elementCount * param.typeSize] = {0};
    uint8_t srcIndexGm[param.elementCount * sizeof(uint32_t)] = {0};
    uint8_t dstValueGm[param.elementCount * param.typeSize] = {0};
    uint8_t dstIndexGm[param.elementCount * sizeof(uint32_t)] = {0};
    param.cal_func(dstValueGm, dstIndexGm, srcValueGm, srcIndexGm, param.elementCount);
    
    for (int32_t i = 0; i < param.elementCount; i++) {
        EXPECT_EQ(dstValueGm[i], 0x00);
        EXPECT_EQ(dstIndexGm[i], 0x00);
    }
}