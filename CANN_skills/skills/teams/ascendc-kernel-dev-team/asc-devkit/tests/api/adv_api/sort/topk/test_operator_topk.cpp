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
#define private public
#define protected public
#include "kernel_operator.h"
#include <iostream>
using namespace std;
using namespace AscendC;

namespace TEST_CASE {
constexpr uint32_t TOPK_CALC_FAC = 16;
constexpr uint32_t TOPK_HALF_SIZE = 2;
constexpr uint32_t TOPK_FLOAT_SIZE = 4;
constexpr uint32_t BINARY_10 = 2;
constexpr uint32_t BINARY_0001 = 4;
constexpr uint32_t BINARY_THIRTYTWO = 2863311530;
constexpr uint32_t BINARY_SIXTEEN = 4369;
constexpr int32_t FIFTEEN = 15;
constexpr int32_t SIXTEEN = 16;
constexpr int32_t SEVENTEEN = 17;
constexpr int32_t THIRTYTWO = 32;
constexpr int32_t TWO = 2;
constexpr int32_t FOUR = 4;
constexpr int32_t FIVE = 5;
constexpr int32_t SIX = 6;
constexpr int32_t SEVEN = 7;
constexpr int32_t EIGHT = 8;
constexpr int32_t NINE = 9;
constexpr int32_t TWELVE = 12;
constexpr int32_t THIRTEEN = 13;
constexpr int32_t TWENTY = 20;
constexpr int32_t TWENTYONE = 21;
constexpr int32_t TWENTYFOUR = 24;
constexpr int32_t TWENTYFIVE = 25;
constexpr int32_t TWENTYEIGHT = 28;
constexpr int32_t TWENTYNINE = 29;
constexpr int32_t THIRTYTHREE = 33;
constexpr int32_t TOPKNSMALL_MASK_BYTES = 64;
constexpr int32_t SORT32_ONE_NUM_BYTES = 8;
constexpr int32_t ONE_BLK_SIZE = 32;
constexpr int32_t PROPOSAL_CONTAIN_NUMBER = 8;
constexpr int32_t MIN_RPSORT16_SIZE = 16;
constexpr int32_t BINARY_SIXTEEN_ONE = 65535;

uint32_t generateBinary10(int k)
{
    uint32_t m = BINARY_10;  // 10
    for (int i = 1; i < k; ++i) {
        m = (m << BINARY_10) | m;
    }
    return m;
}
uint32_t generateBinary01(int k)
{
    uint32_t m = 1;  // 01
    for (int i = 1; i < k; ++i) {
        m = (m << BINARY_10) | m;
    }
    return m;
}

uint16_t generateBinary0001(int k)
{
    uint16_t m = 1;  // 0001
    for (int i = 1; i < k; ++i) {
        m = (m << BINARY_0001) | m;
    }
    return m;
}

void GetVreduceMask(const int32_t k, const uint32_t dataTypeSize, TopkTiling &topKTiling)
{
    if (k < SEVENTEEN) {
        topKTiling.vreduceIdxMask0 = (generateBinary10(k));
    } else if (k <= THIRTYTWO) {
        topKTiling.vreduceIdxMask0 = (BINARY_THIRTYTWO);
        topKTiling.vreduceIdxMask1 = (generateBinary10(k - SIXTEEN));
    }
    if (dataTypeSize == TOPK_FLOAT_SIZE) {
        if (k < SEVENTEEN) {
            topKTiling.vreduceValMask0 = (generateBinary01(k));
        } else if (k <= THIRTYTWO) {
            topKTiling.vreduceValMask0 = (generateBinary01(SIXTEEN));
            topKTiling.vreduceValMask1 = (generateBinary01(k - SIXTEEN));
        }
    } else {
        topKTiling.vreducehalfValMask0 = ((k < FIVE) ? generateBinary0001(k) : BINARY_SIXTEEN);
        topKTiling.vreducehalfValMask1 = ((k >= FIVE) ? (generateBinary0001((k >= NINE) ? FOUR : (k - FOUR))) : 0);
        topKTiling.vreducehalfValMask2 = (
            (k >= NINE) ? (generateBinary0001((k >= THIRTEEN) ? FOUR : (k - EIGHT))) : 0);
        topKTiling.vreducehalfValMask3 = (
            (k >= THIRTEEN) ? (generateBinary0001((k >= SEVENTEEN) ? FOUR : (k - TWELVE))) : 0);
        topKTiling.vreducehalfValMask4 = (
            (k >= SEVENTEEN) ? (generateBinary0001((k >= TWENTYONE) ? FOUR : (k - SIXTEEN))) : 0);
        topKTiling.vreducehalfValMask5 = (
            (k >= TWENTYONE) ? (generateBinary0001((k >= TWENTYFIVE) ? FOUR : (k - TWENTY))) : 0);
        topKTiling.vreducehalfValMask6 = (
            (k >= TWENTYFIVE) ? (generateBinary0001((k >= TWENTYNINE) ? FOUR : (k - TWENTYFOUR))) : 0);
        topKTiling.vreducehalfValMask7 = (
            (k >= TWENTYNINE) ? (generateBinary0001((k >= THIRTYTHREE) ? FOUR : (k - TWENTYEIGHT))) : 0);
    }
}

void SetTopkNormalVal(const int32_t inner, const int32_t outter, const int32_t k, const uint32_t dataTypeSize,
    const bool isInitIndex, const int32_t kAlignFourBytesTmp, TopkTiling &topKTiling)
{
    int32_t kAlignTwoBytesTmp = ((k + FIFTEEN) / SIXTEEN) * SIXTEEN;
    // inner * 16 / sizeof(float) = inner * 4
    topKTiling.tmpLocalSize = (FOUR * inner);
    topKTiling.innerDataSize = (inner * TWO);
    int32_t copyUbufToUbufBlockCount = inner * 2 * TOPK_FLOAT_SIZE / ONE_BLK_SIZE;
    topKTiling.copyUbToUbBlockCount = (copyUbufToUbufBlockCount);
    topKTiling.sortRepeat = (inner / THIRTYTWO);
    topKTiling.mrgSortRepeat = (inner / FOUR);
    topKTiling.kAlignTwoBytes = (kAlignTwoBytesTmp);
    int32_t maskOffsetFloat = outter * kAlignFourBytesTmp;
    int32_t maskOffsetHalf = outter * kAlignTwoBytesTmp;
    topKTiling.maskOffset = (maskOffsetFloat);
    topKTiling.maskVreducev2FourBytes = (TWO * k);
    topKTiling.maskVreducev2TwoBytes = (FOUR * k);
    // float: 1 * 2
    topKTiling.mrgSortSrc1offset = (TWO);
    // float: 2 * 2
    topKTiling.mrgSortSrc2offset = (FOUR);
    // float: 3 * 2
    topKTiling.mrgSortSrc3offset = (SIX);
    topKTiling.mrgSortTwoQueueSrc1Offset = (TWO);
    topKTiling.mrgFourQueueTailPara1 = (inner * TWO);
    topKTiling.mrgFourQueueTailPara2 = sizeof(float) / dataTypeSize;
    if (dataTypeSize == TOPK_HALF_SIZE) {
        // inner * 16 / sizeof(half) = inner * 8
        topKTiling.tmpLocalSize = (EIGHT * inner);
        topKTiling.maskOffset = (maskOffsetHalf);
        topKTiling.innerDataSize = (inner * FOUR);
        // 2 * 2
        topKTiling.mrgSortSrc1offset = (FOUR);
        // 2 * 2 * 2
        topKTiling.mrgSortSrc2offset = (EIGHT);
        // 3 * 2 * 2
        topKTiling.mrgSortSrc3offset = (TWELVE);
        topKTiling.mrgSortTwoQueueSrc1Offset = (FOUR);
    }
    if (!isInitIndex) {
        topKTiling.tmpLocalSize = (FOUR * inner + inner);
        topKTiling.srcIndexOffset = (FOUR * inner);
        if (dataTypeSize == TOPK_HALF_SIZE) {
            topKTiling.tmpLocalSize = (EIGHT * inner + inner * TWO);
            topKTiling.srcIndexOffset = (EIGHT * inner);
        }
    }
}

void SetTopkNSmallVal(const int32_t inner, const int32_t outter, const int32_t k, const uint32_t dataTypeSize,
    const bool isInitIndex, TopkTiling &topKTiling, const bool isLargest = false)
{
    int32_t innerDataSize = inner * outter * (SORT32_ONE_NUM_BYTES / dataTypeSize);
    topKTiling.innerDataSize = (innerDataSize);
    topKTiling.maskOffset = (outter * k);
    int32_t generateNegativeValSize = 0;;
    if (!isLargest) {
        generateNegativeValSize = inner * outter;
    }
    int32_t MrgSrc1MaskSizeOffset = (innerDataSize) + generateNegativeValSize;
    topKTiling.topkMrgSrc1MaskSizeOffset = (MrgSrc1MaskSizeOffset);
    topKTiling.tmpLocalSize = (MrgSrc1MaskSizeOffset + TOPKNSMALL_MASK_BYTES / dataTypeSize);
    if (!isInitIndex) {
        int32_t srcIndexLocalOffset = MrgSrc1MaskSizeOffset + TOPKNSMALL_MASK_BYTES / dataTypeSize;
        // Space required for generate indexs: inner * outter * TOPK_FLOAT_SIZE / dataTypeSize
        topKTiling.tmpLocalSize = (srcIndexLocalOffset + inner * outter * TOPK_FLOAT_SIZE / dataTypeSize);
        topKTiling.topkNSmallSrcIndexOffset = (srcIndexLocalOffset);
    }
    GetVreduceMask(k, dataTypeSize, topKTiling);
}

bool TopKTilingFunc(const int32_t inner, const int32_t outter, const int32_t k,
    const uint32_t dataTypeSize, const bool isInitIndex, enum TopKMode topkMode, TopkTiling &topKTiling,
    const bool isLargest = false)
{
    if (dataTypeSize == 0) {
        return false;
    }
    topKTiling.allDataSize = (inner * outter);
    int32_t kAlignFourBytesTmp = ((k + SEVEN) / EIGHT) * EIGHT;
    topKTiling.kAlignFourBytes = (kAlignFourBytesTmp);
    // TOPK_NORMAL = 0, TOPK_NSMALL = 1
    if (topkMode == TopKMode::TOPK_NORMAL) {
        SetTopkNormalVal(inner, outter, k, dataTypeSize, isInitIndex, kAlignFourBytesTmp, topKTiling);
    } else {
        SetTopkNSmallVal(inner, outter, k, dataTypeSize, isInitIndex, topKTiling, isLargest);
    }
    return true;
}
}  // namespace TEST_CASE

template <typename T, typename U, bool isInitIndex = false, bool isHasfinish = false, bool isReuseSrc = false,
    bool isBasicBlock = false, enum TopKMode topkMode = AscendC::TopKMode::TOPK_NORMAL>
class KernelTopK {
public:
    __aicore__ inline KernelTopK()
    {}
    __aicore__ inline void Init(GM_ADDR srcGmValue, GM_ADDR srcGmIndex, GM_ADDR dstGmValue, GM_ADDR dstGmIndex,
        GM_ADDR finishGm, int32_t kGm, TopKInfo infoGm, bool isLargestGm)
    {

        k = kGm;
        // cal kPad
        if (sizeof(T) == sizeof(float)) {
            kPad = (k + 7) / 8 * 8;
        } else {
            kPad = (k + 15) / 16 * 16;
        }

        kPadIndex = (k + 7) / 8 * 8;
        isLargest = isLargestGm;
        topKInfo.outter = infoGm.outter;
        topKInfo.inner = infoGm.inner;
        topKInfo.n = infoGm.n;

        inDataSize = topKInfo.inner * topKInfo.outter;
        outValueDataSize = kPad * topKInfo.outter;
        outIndexDataSize = kPadIndex * topKInfo.outter;

        inputdexDataSize = topKInfo.inner;
        if (topkMode == AscendC::TopKMode::TOPK_NSMALL) {
            inputdexDataSize = inDataSize;
        }

        srcGlobal1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGmValue), inDataSize);
        srcGlobal2.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(srcGmIndex), inputdexDataSize);
        srcGlobal3.SetGlobalBuffer(reinterpret_cast<__gm__ bool *>(finishGm), topKInfo.outter);
        dstGlobal1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGmValue), outValueDataSize);
        dstGlobal2.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(dstGmIndex), outIndexDataSize);

        pipe.InitBuffer(inQueueX1, 1, inDataSize * sizeof(T));
        pipe.InitBuffer(inQueueX2, 1, inputdexDataSize * sizeof(U));

        finishLocalBytes = topKInfo.outter * sizeof(bool);
        if (finishLocalBytes % 32 != 0) {
            finishLocalBytes = (finishLocalBytes + 31) / 32 * 32;
        }
        pipe.InitBuffer(inQueueX3, 1, finishLocalBytes);
        pipe.InitBuffer(outQueueY1, 1, outValueDataSize * sizeof(T));
        pipe.InitBuffer(outQueueY2, 1, outIndexDataSize * sizeof(U));
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
        LocalTensor<T> srcLocalValue = inQueueX1.AllocTensor<T>();
        LocalTensor<U> srcLocalIndex = inQueueX2.AllocTensor<U>();
        LocalTensor<bool> srcLocalFinish = inQueueX3.AllocTensor<bool>();
        DataCopy(srcLocalValue, srcGlobal1, inDataSize);
        DataCopy(srcLocalIndex, srcGlobal2, inputdexDataSize);
        DataCopy(srcLocalFinish, srcGlobal3, finishLocalBytes);

        inQueueX1.EnQue(srcLocalValue);
        inQueueX2.EnQue(srcLocalIndex);
        inQueueX3.EnQue(srcLocalFinish);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> dstLocalValue = outQueueY1.AllocTensor<T>();
        LocalTensor<U> dstLocalIndex = outQueueY2.AllocTensor<U>();

        LocalTensor<T> srcLocalValue = inQueueX1.DeQue<T>();
        LocalTensor<U> srcLocalIndex = inQueueX2.DeQue<U>();
        LocalTensor<bool> srcLocalFinish = inQueueX3.DeQue<bool>();

        TopkTiling topKTiling;
        TEST_CASE::TopKTilingFunc(topKInfo.inner, topKInfo.outter, k, sizeof(T), isInitIndex, topkMode, topKTiling);
        TopK<T, isInitIndex, isHasfinish, isReuseSrc, topkMode>(dstLocalValue,
            dstLocalIndex,
            srcLocalValue,
            srcLocalIndex,
            srcLocalFinish,
            k,
            topKTiling,
            topKInfo,
            isLargest);
        outQueueY1.EnQue<T>(dstLocalValue);
        outQueueY2.EnQue<U>(dstLocalIndex);

        inQueueX1.FreeTensor(srcLocalValue);
        inQueueX2.FreeTensor(srcLocalIndex);
        inQueueX3.FreeTensor(srcLocalFinish);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocalValue = outQueueY1.DeQue<T>();
        LocalTensor<U> dstLocalIndex = outQueueY2.DeQue<U>();

        DataCopy(dstGlobal1, dstLocalValue, outValueDataSize);
        DataCopy(dstGlobal2, dstLocalIndex, outIndexDataSize);
        outQueueY1.FreeTensor(dstLocalValue);
        outQueueY2.FreeTensor(dstLocalIndex);
    }

private:
    GlobalTensor<T> srcGlobal1;
    GlobalTensor<U> srcGlobal2;
    GlobalTensor<bool> srcGlobal3;
    GlobalTensor<T> dstGlobal1;
    GlobalTensor<U> dstGlobal2;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX1;
    TQue<TPosition::VECIN, 1> inQueueX2;
    TQue<TPosition::VECIN, 1> inQueueX3;
    TQue<TPosition::VECOUT, 1> outQueueY1;
    TQue<TPosition::VECOUT, 1> outQueueY2;

    uint32_t inDataSize = 0;
    uint32_t inputdexDataSize = 0;
    uint32_t inputdexBytes = 0;
    uint32_t finishLocalBytes;
    uint32_t outValueDataSize = 0;
    uint32_t outIndexDataSize = 0;
    int32_t k;
    int32_t kPad;
    int32_t kPadIndex;
    bool isLargest = true;
    TopKInfo topKInfo;
};

template <typename T, typename U, bool isInitIndex = false, bool isHasfinish = false, bool isReuseSrc = false,
    bool isBasicBlock = false, int32_t topkMode>
__aicore__ void main_TopK_test(GM_ADDR srcGmValue, GM_ADDR srcGmIndex, GM_ADDR dstGmValue, GM_ADDR dstGmIndex,
    GM_ADDR finishGm, int32_t kGm, TopKInfo infoGm, bool isLargest)
{
    if (topkMode == 0) {
        KernelTopK<T, U, isInitIndex, isHasfinish, isReuseSrc, isBasicBlock, AscendC::TopKMode::TOPK_NORMAL> op;
        op.Init(srcGmValue, srcGmIndex, dstGmValue, dstGmIndex, finishGm, kGm, infoGm, isLargest);
        op.Process();
    } else {
        KernelTopK<T, U, isInitIndex, isHasfinish, isReuseSrc, isBasicBlock, AscendC::TopKMode::TOPK_NSMALL> op;
        op.Init(srcGmValue, srcGmIndex, dstGmValue, dstGmIndex, finishGm, kGm, infoGm, isLargest);
        op.Process();
    }
}

struct TopKTestParams {
    int32_t k;
    TopKInfo shapeInfo;
    uint32_t dataTypeSize;
    uint32_t indexTypeSize;
    bool islargest;
    void (*calFunc)(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, int32_t, TopKInfo, bool);
};

class TopKTestSuite : public testing::Test, public testing::WithParamInterface<TopKTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TopKTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "TopKTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp()
    {}
    virtual void TearDown()
    {}
};

// inner 32 Multiple
INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_TopK, TopKTestSuite,
    ::testing::Values(
    TopKTestParams { 1,  {1, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 1>},
    TopKTestParams { 5,  {1, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 1>},
    TopKTestParams { 8, {1, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 1>},
    TopKTestParams { 16, {1, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 1>},
    TopKTestParams { 31, {1, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 1>},
    TopKTestParams { 32, {1, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 1>},
    TopKTestParams { 5,  {4, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 1>},
    TopKTestParams { 8,  {4, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 1>},
    TopKTestParams { 5,  {4, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, true, 1>},
    TopKTestParams { 8,  {4, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, true, 1>},
    TopKTestParams { 5,  {4, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, true, false, 1>},
    TopKTestParams { 8,  {4, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, true, false, 1>},
    TopKTestParams { 16,  {4, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, true, false, 1>},
    TopKTestParams { 5,  {4, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, true, 1>},
    TopKTestParams { 8,  {4, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, true, 1>},
    TopKTestParams { 16,  {4, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 1>},
    TopKTestParams { 16,  {4, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, true, 1>},
    TopKTestParams { 1,  {1, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 0>},
    TopKTestParams { 8,  {1, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 0>},
    TopKTestParams { 16, {1, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 0>},
    TopKTestParams { 32, {1, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 0>},
    TopKTestParams { 8,  {1, 32, 8}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 0>},
    TopKTestParams { 8,  {1, 32, 16}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 0>},
    TopKTestParams { 8,  {1, 32, 32}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 0>},
    TopKTestParams { 64, {1, 64, 64}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 0>},
    TopKTestParams { 64, {1, 96, 96}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 0>},
    TopKTestParams { 64, {1, 128, 128}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 0>},
    TopKTestParams { 64, {1, 160, 160}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 0>},
    TopKTestParams { 64, {1, 192, 192}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 0>},
    TopKTestParams { 64, {1, 2048, 2048}, sizeof(half), sizeof(int32_t), true, main_TopK_test<half, int32_t, false, false, false, false, 0>}
));

TEST_P(TopKTestSuite, TopKTestCase)
{
    auto param = GetParam();
    uint32_t inDataSize = param.shapeInfo.outter * param.shapeInfo.inner;
    uint8_t inputGmValue[inDataSize * param.dataTypeSize]{0x00};
    uint8_t inputGmIndex[inDataSize * param.indexTypeSize]{0x00};
    uint8_t finishGm[param.shapeInfo.outter]{0x00};
    uint32_t outDataSize = param.shapeInfo.outter * param.k;
    uint8_t outputGmValue[outDataSize * param.dataTypeSize]{0x00};
    uint8_t outputGmIndex[outDataSize * param.indexTypeSize]{0x00};
    bool isLargest = param.islargest;
    param.calFunc(
        inputGmValue, inputGmIndex, outputGmValue, outputGmIndex, finishGm, param.k, param.shapeInfo, isLargest);
    for (int32_t i = 0; i < outDataSize; i++) {
        EXPECT_EQ(outputGmValue[i], 0x00);
    }
}