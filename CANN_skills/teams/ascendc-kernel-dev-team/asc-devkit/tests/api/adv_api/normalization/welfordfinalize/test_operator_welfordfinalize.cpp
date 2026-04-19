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
constexpr uint32_t WELFORDFINALIZE_BASICBLOCK_UNIT = 256 / sizeof(float);

void GetWelfordFinalizeMaxMinTmpSize(const uint32_t abLength, const uint32_t typeSize, const bool isReuseSource,
                                         uint32_t &maxValue, uint32_t &minValue)
{
    (void)isReuseSource;
    (void)typeSize;

    // 0x4 indicates reserving four buffers for the calculate/storage of mean and variance
    minValue = WELFORDFINALIZE_BASICBLOCK_UNIT * 0x4 * sizeof(float);

    if (abLength <= WELFORDFINALIZE_BASICBLOCK_UNIT) {
        maxValue = minValue;
    } else {
        maxValue = (WELFORDFINALIZE_BASICBLOCK_UNIT *0x2 + abLength * 0x2) * sizeof(float);
    }
}

}  // namespace TEST_CASE

template <typename dataType, bool isReuseSource = false, bool isCounts = false, bool sharedTmp = false>
class KernelWelfordFinalize {
public:
    __aicore__ inline KernelWelfordFinalize()
    {}
    __aicore__ inline void Init(GM_ADDR inputMean_gm, GM_ADDR inputVariance_gm, GM_ADDR counts_gm, GM_ADDR outputMean_gm,
        GM_ADDR outputVariance_gm, uint32_t rnLength, uint32_t abLength, uint32_t head, uint32_t headLength, uint32_t tail, uint32_t tailLength)
    {
        this->rnLength = rnLength;
        this->abLength = abLength;
        this->head = head;
        this->headLength = headLength;
        this->tail = tail;
        this->tailLength = tailLength;
        if (tailLength == 0){
            this->rLength = rnLength * abLength;
        } else {
            this->rLength = head * headLength + tail * tailLength;
        }
        this->abRec = 1.0f / abLength;
        this->rRec = 1.0f / rLength;
        this->outLength = 8;
 
        inputMean_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputMean_gm), abLength);
        inputVariance_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputVariance_gm), abLength);
        inputcounts_global.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(counts_gm), abLength);
        outputMean_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(outputMean_gm), outLength);
        outputVariance_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(outputVariance_gm), outLength);

        pipe.InitBuffer(inQueueMean, 1, abLength * sizeof(dataType));
        pipe.InitBuffer(inQueueVariance, 1, abLength * sizeof(dataType));
        pipe.InitBuffer(inQueueCounts, 1, abLength * sizeof(int32_t));
        pipe.InitBuffer(outQueueMean, 1, outLength * sizeof(dataType));
        pipe.InitBuffer(outQueueVariance, 1, outLength * sizeof(dataType));

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
        LocalTensor<dataType> inputMeanLocal = inQueueMean.AllocTensor<dataType>();
        LocalTensor<dataType> inputVarianceLocal = inQueueVariance.AllocTensor<dataType>();
        LocalTensor<int32_t> inputCountsLocal = inQueueCounts.AllocTensor<int32_t>();

        DataCopy(inputMeanLocal, inputMean_global, abLength);
        DataCopy(inputVarianceLocal, inputVariance_global, abLength);
        DataCopy(inputCountsLocal, inputcounts_global, abLength);

        inQueueMean.EnQue(inputMeanLocal);
        inQueueVariance.EnQue(inputVarianceLocal);
        inQueueCounts.EnQue(inputCountsLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<dataType> inputMeanLocal = inQueueMean.DeQue<dataType>();
        LocalTensor<dataType> inputVarianceLocal = inQueueVariance.DeQue<dataType>();
        LocalTensor<int32_t> inputCountsLocal = inQueueCounts.DeQue<int32_t>();

        LocalTensor<dataType> meanLocal = outQueueMean.AllocTensor<dataType>();
        LocalTensor<dataType> varianceLocal = outQueueVariance.AllocTensor<dataType>();

        uint32_t minvalue = 0;
        uint32_t maxValue = 0;
        TEST_CASE::GetWelfordFinalizeMaxMinTmpSize(abLength, sizeof(dataType), isReuseSource, minvalue, maxValue);
        pipe.InitBuffer(sharedTmpBuffer, maxValue);
        LocalTensor<uint8_t> tmpLocalTensor = sharedTmpBuffer.Get<uint8_t>();

        struct WelfordFinalizePara para = {rnLength, abLength, head, headLength, tail, tailLength, abRec, rRec};

        if constexpr (isCounts) {
            if constexpr (sharedTmp) {
                WelfordFinalize<false>(meanLocal, varianceLocal, inputMeanLocal, inputVarianceLocal, inputCountsLocal, tmpLocalTensor, para);
            } else {
                WelfordFinalize<false>(meanLocal, varianceLocal, inputMeanLocal, inputVarianceLocal, inputCountsLocal, para);
            }
        } else {
            if constexpr (sharedTmp) {
                WelfordFinalize<false>(meanLocal, varianceLocal, inputMeanLocal, inputVarianceLocal, tmpLocalTensor, para);
            } else {
                WelfordFinalize<false>(meanLocal, varianceLocal, inputMeanLocal, inputVarianceLocal, para);
            }
        }
        
        outQueueMean.EnQue<dataType>(meanLocal);
        outQueueVariance.EnQue<dataType>(varianceLocal);

        inQueueMean.FreeTensor(inputMeanLocal);
        inQueueVariance.FreeTensor(inputVarianceLocal);
        inQueueCounts.FreeTensor(inputCountsLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<dataType> meanLocal = outQueueMean.DeQue<dataType>();
        LocalTensor<dataType> varianceLocal = outQueueVariance.DeQue<dataType>();

        DataCopy(outputMean_global, meanLocal, outLength);
        DataCopy(outputVariance_global, varianceLocal, outLength);

        outQueueMean.FreeTensor(meanLocal);
        outQueueVariance.FreeTensor(varianceLocal);
    }

private:
    GlobalTensor<dataType> inputMean_global;
    GlobalTensor<dataType> inputVariance_global;
    GlobalTensor<int32_t> inputcounts_global;
    GlobalTensor<dataType> outputMean_global;
    GlobalTensor<dataType> outputVariance_global;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueMean;
    TQue<TPosition::VECIN, 1> inQueueVariance;
    TQue<TPosition::VECIN, 1> inQueueCounts;
    TQue<TPosition::VECOUT, 1> outQueueMean;
    TQue<TPosition::VECOUT, 1> outQueueVariance;
    TBuf<TPosition::VECCALC> sharedTmpBuffer;

    uint32_t rnLength;
    uint32_t abLength;
    uint32_t rLength;
    uint32_t head;
    uint32_t headLength;
    uint32_t tail;
    uint32_t tailLength;
    uint32_t outLength;
    float abRec;
    float rRec;
    bool inplace;

    uint32_t stackBufferSize = 0;
};

template <typename dataType, bool isCounts = false, bool sharedTmp = false>
__aicore__ void kernel_WelfordFinalize_test(GM_ADDR inMeanGm, GM_ADDR inVarGm, GM_ADDR countsGm, GM_ADDR outMeanGm,
    GM_ADDR outVarGm, uint32_t rnLength, uint32_t abLength, uint32_t head, uint32_t headLength, uint32_t tail, uint32_t tailLength)
{
    KernelWelfordFinalize<dataType, isCounts, sharedTmp> op;
    op.Init(inMeanGm, inVarGm, countsGm, outMeanGm, outVarGm, rnLength, abLength, head, headLength, tail, tailLength);
    op.Process();
}

struct WelfordFinalizeTestParams {
    uint32_t rnLength;
    uint32_t abLength;
    uint32_t head;
    uint32_t headLength;
    uint32_t tail;
    uint32_t tailLength;

    void (*calFunc)(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
};

class WelfordFinalizeTestSuite : public testing::Test, public testing::WithParamInterface<WelfordFinalizeTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "WelfordFinalizeTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "WelfordFinalizeTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp()
    {}
    virtual void TearDown()
    {}
};

// 1. with tail block;
// 2. there is counts;
// 3. there is buffer constraint;
INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_WelfordFinalize, WelfordFinalizeTestSuite,
   ::testing::Values(
   WelfordFinalizeTestParams { 4, 32, 4, 32, 4, 0, kernel_WelfordFinalize_test<float, false, false> }, // !1 + !2 + !3
   WelfordFinalizeTestParams { 4, 64, 4, 64, 4, 0, kernel_WelfordFinalize_test<float, false, false> }, // !1 + !2 + !3
   WelfordFinalizeTestParams { 4, 4096, 4, 4096, 4, 0, kernel_WelfordFinalize_test<float, false, false> }, // !1 + !2 + !3
   WelfordFinalizeTestParams { 4, 4096, 4, 3000, 3, 1096, kernel_WelfordFinalize_test<float, false, false> },// 1 + !2 + !3
   WelfordFinalizeTestParams { 4, 10240, 4, 10240, 4, 0, kernel_WelfordFinalize_test<float, false, true> }, // !1 + !2 + 3
   WelfordFinalizeTestParams { 4, 10240, 4, 10000, 3, 240, kernel_WelfordFinalize_test<float, false, true> }, // 1 + !2 + 3
   WelfordFinalizeTestParams { 4, 4096, 4, 4096, 4, 0, kernel_WelfordFinalize_test<float, true, false> }, // !1 + 2 + !3
   WelfordFinalizeTestParams { 4, 4096, 4, 3000, 3, 1096, kernel_WelfordFinalize_test<float, true, false> }, // 1 + 2 + !3
   WelfordFinalizeTestParams { 4, 10240, 4, 10240, 4, 0, kernel_WelfordFinalize_test<float, true, true> }, // !1 + 2 + 3
   WelfordFinalizeTestParams { 4, 10240, 4, 10000, 3, 240, kernel_WelfordFinalize_test<float, true, true> } // 1 + 2 + 3
));

TEST_P(WelfordFinalizeTestSuite, WelfordFinalizeTestCase)
{
   auto param = GetParam();
   uint32_t srcSize = param.abLength;
   uint32_t outSize = 8;
   uint8_t inMeanGm[srcSize * sizeof(float)]{0x00};
   uint8_t inVarGm[srcSize * sizeof(float)]{0x00};
   uint8_t countsGm[srcSize * sizeof(int32_t)]{0x00};
   uint8_t outMeanGm[outSize * sizeof(float)]{0x00};
   uint8_t outVarGm[outSize * sizeof(float)]{0x00};
   param.calFunc(inMeanGm, inVarGm, countsGm, outMeanGm, outVarGm, param.rnLength, param.abLength, param.head, param.headLength, param.tail, param.tailLength);
   for (int32_t i = 0; i < outSize; i++) {
       EXPECT_EQ(outMeanGm[i], 0x00);
       EXPECT_EQ(outVarGm[i], 0x00);
   }
}