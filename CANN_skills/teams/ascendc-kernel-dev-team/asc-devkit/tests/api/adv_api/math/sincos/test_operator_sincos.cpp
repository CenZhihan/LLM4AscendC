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

using namespace AscendC;

template <typename srcType, uint32_t calCount, uint32_t dataSize, uint32_t apiMode>
class KernelSinCos {
public:
    __aicore__ inline KernelSinCos()
    {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst0_gm, GM_ADDR dst1_gm)
    {
        src_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src_gm), dataSize);
        dst0_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dst0_gm), dataSize);
        dst1_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dst1_gm), dataSize);

        pipe.InitBuffer(inQueueX, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(tmpQueue, 1, dataSize * sizeof(uint8_t));
        pipe.InitBuffer(outQueueX, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(outQueueY, 1, dataSize * sizeof(srcType));
    }
    __aicore__ inline void Process()
    {
        AscendCUtils::SetOverflow(1);
        CopyIn();
        Compute();
        CopyOut();
        AscendCUtils::SetOverflow(0);
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<srcType> srcLocal = inQueueX.AllocTensor<srcType>();
        LocalTensor<uint8_t> tmpBufLocal = tmpQueue.AllocTensor<uint8_t>();
        DataCopy(srcLocal, src_global, dataSize);
        inQueueX.EnQue(srcLocal);
        tmpQueue.EnQue(tmpBufLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<srcType> dstLocalX = outQueueX.AllocTensor<srcType>();
        LocalTensor<srcType> dstLocalY = outQueueY.AllocTensor<srcType>();

        LocalTensor<srcType> srcLocal = inQueueX.DeQue<srcType>();
        LocalTensor<uint8_t> tmpBuf = tmpQueue.DeQue<uint8_t>();
        srcType zero = 0;
        Duplicate(dstLocalX, zero, dataSize);
        Duplicate(dstLocalY, zero, dataSize);
        uint32_t count = (uint32_t)calCount;

        if constexpr (apiMode == 1) {
            SinCos(dstLocalX, dstLocalY, srcLocal, count);
        } else if constexpr (apiMode == 0) {
            SinCos(dstLocalX, dstLocalY, srcLocal, tmpBuf, count);
        }

        outQueueX.EnQue<srcType>(dstLocalX);
        outQueueY.EnQue<srcType>(dstLocalY);
        inQueueX.FreeTensor(srcLocal);
        tmpQueue.FreeTensor(tmpBuf);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<srcType> dstLocalX = outQueueX.DeQue<srcType>();
        DataCopy(dst0_global, dstLocalX, dataSize);
        outQueueX.FreeTensor(dstLocalX);

        LocalTensor<srcType> dstLocalY = outQueueY.DeQue<srcType>();
        DataCopy(dst1_global, dstLocalY, dataSize);
        outQueueY.FreeTensor(dstLocalY);
    }

private:
    GlobalTensor<srcType> src_global;
    GlobalTensor<srcType> dst0_global;
    GlobalTensor<srcType> dst1_global;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> tmpQueue;
    TQue<QuePosition::VECOUT, 1> outQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
};

template <typename srcType, uint32_t calCount, uint32_t dataSize, uint32_t apiMode>
__aicore__ void kernel_sincos_operator(GM_ADDR src_gm, GM_ADDR dst0_gm, GM_ADDR dst1_gm)
{
    KernelSinCos<srcType, calCount, dataSize, apiMode> op;
    op.Init(src_gm, dst0_gm, dst1_gm);
    op.Process();
}

struct SinCosTestParams {
    uint32_t dataTypeSize;
    uint32_t inDataSize;
    void (*calFunc)(GM_ADDR, GM_ADDR, GM_ADDR);
};

class AdvanceSinCosTestSuite : public testing::TestWithParam<SinCosTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ADVANCE_API_SINCOS, AdvanceSinCosTestSuite,
    ::testing::Values(
        SinCosTestParams {4, 32, kernel_sincos_operator<float, 32, 32, 0>},
        SinCosTestParams {4, 4096, kernel_sincos_operator<float, 32, 4096, 1>},
        SinCosTestParams {4, 32, kernel_sincos_operator<float, 32, 32, 0>},
        SinCosTestParams {4, 4096, kernel_sincos_operator<float, 32, 4096, 1>},
        SinCosTestParams {2, 32, kernel_sincos_operator<half, 32, 32, 0>},
        SinCosTestParams {2, 4096, kernel_sincos_operator<half, 32, 4096, 1>},
        SinCosTestParams {2, 32, kernel_sincos_operator<half, 32, 32, 0>},
        SinCosTestParams {2, 4096, kernel_sincos_operator<half, 32, 4096, 1>}
    )
);

TEST_P(AdvanceSinCosTestSuite, kernel_sincos_operator)
{
    auto param = GetParam();
    uint32_t dataTypeSize = param.dataTypeSize;
    uint32_t inDataSize = param.inDataSize;
    uint8_t inputGm[inDataSize * dataTypeSize] = {1};
    uint8_t outputGm0[inDataSize * dataTypeSize] = {0};
    uint8_t outputGm1[inDataSize * dataTypeSize] = {0};
    param.calFunc(inputGm, outputGm0, outputGm1);
    for (int32_t i = 0; i < inDataSize; i++) {
        EXPECT_EQ(outputGm0[i], 0x00);
        EXPECT_EQ(outputGm1[i], 0x00);
    }
}
