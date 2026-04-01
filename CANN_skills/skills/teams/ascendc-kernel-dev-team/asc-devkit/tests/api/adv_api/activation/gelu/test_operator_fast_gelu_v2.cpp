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

using namespace std;
using namespace AscendC;

template <typename srcType> class KernelFastGeluV2 {
public:
    __aicore__ inline KernelFastGeluV2() {}
    __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t inputSize)
    {
        dataSize = inputSize;

        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType*>(srcGm), dataSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType*>(dstGm), dataSize);

        pipe.InitBuffer(inQueueX, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(srcType));
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
        LocalTensor<srcType> srcLocal = inQueueX.AllocTensor<srcType>();

        DataCopy(srcLocal, srcGlobal, dataSize);

        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();

        LocalTensor<srcType> srcLocal = inQueueX.DeQue<srcType>();

        FasterGeluV2(dstLocal, srcLocal, dataSize);

        outQueue.EnQue<srcType>(dstLocal);

        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        DataCopy(dstGlobal, dstLocal, dataSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<srcType> srcGlobal;
    GlobalTensor<srcType> dstGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECOUT, 1> outQueue;

    uint32_t dataSize = 0;
};

template <typename dataType>
__aicore__ void main_FastGeluV2_test(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t dataSize)
{
    KernelFastGeluV2<dataType> op;
    op.Init(srcGm, dstGm, dataSize);
    op.Process();
}

struct FastGeluV2TestParams {
    uint32_t dataSize;
    uint32_t dataTypeSize;
    void (*calFunc)(uint8_t*, uint8_t*, uint32_t);
};

class FastGeluV2TestSuite : public testing::Test, public testing::WithParamInterface<FastGeluV2TestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FastGeluV2TestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "FastGeluV2TestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {
        AscendC::SetGCoreType(2);
    }
    virtual void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_FastGeluV2, FastGeluV2TestSuite,
    ::testing::Values(
    FastGeluV2TestParams { 32,       sizeof(half),  main_FastGeluV2_test<half> },
    FastGeluV2TestParams { 64,       sizeof(half),  main_FastGeluV2_test<half> },
    FastGeluV2TestParams { 256,      sizeof(half),  main_FastGeluV2_test<half> },
    FastGeluV2TestParams { 512,      sizeof(half),  main_FastGeluV2_test<half> },
    FastGeluV2TestParams { 1024,     sizeof(half),  main_FastGeluV2_test<half> },
    FastGeluV2TestParams { 2048,     sizeof(half),  main_FastGeluV2_test<half> },
    FastGeluV2TestParams { 4096,     sizeof(half),  main_FastGeluV2_test<half> },
    FastGeluV2TestParams { 8192,     sizeof(half),  main_FastGeluV2_test<half> },
    FastGeluV2TestParams { 16384,    sizeof(half),  main_FastGeluV2_test<half> },
    FastGeluV2TestParams { 32768,    sizeof(half),  main_FastGeluV2_test<half> },
    FastGeluV2TestParams { 32,       sizeof(float), main_FastGeluV2_test<float> },
    FastGeluV2TestParams { 64,       sizeof(float), main_FastGeluV2_test<float> },
    FastGeluV2TestParams { 256,      sizeof(float), main_FastGeluV2_test<float> },
    FastGeluV2TestParams { 512,      sizeof(float), main_FastGeluV2_test<float> },
    FastGeluV2TestParams { 1024,     sizeof(float), main_FastGeluV2_test<float> },
    FastGeluV2TestParams { 2048,     sizeof(float), main_FastGeluV2_test<float> },
    FastGeluV2TestParams { 4096,     sizeof(float), main_FastGeluV2_test<float> },
    FastGeluV2TestParams { 8192,     sizeof(float), main_FastGeluV2_test<float> },
    FastGeluV2TestParams { 16384,    sizeof(float), main_FastGeluV2_test<float> }
));

TEST_P(FastGeluV2TestSuite, FastGeluV2TestCase)
{
    auto param = GetParam();

    uint8_t inputGm[param.dataSize * param.dataTypeSize] { 0x00 };
    uint8_t outputGm[param.dataSize * param.dataTypeSize] { 0x00 };

    param.calFunc(inputGm, outputGm, param.dataSize);

    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(outputGm[i], 0x00);
    }
}