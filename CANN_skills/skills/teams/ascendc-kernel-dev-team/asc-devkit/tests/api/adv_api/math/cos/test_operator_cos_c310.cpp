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

template <int size = 0> struct GetCosAlgo {
    static constexpr AscendC::CosAlgo value = AscendC::CosAlgo::POLYNOMIAL_APPROXIMATION;
};

template <> struct GetCosAlgo<1> {
    static constexpr AscendC::CosAlgo value = AscendC::CosAlgo::RADIAN_REDUCTION;
};

template <typename srcType, uint32_t calCount, uint32_t dataSize, uint32_t apiMode, uint32_t algoMode = 0>
class KernelCos {
public:
    __aicore__ inline KernelCos()
    {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm)
    {
        src_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src_gm), dataSize);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dst_gm), dataSize);

        pipe.InitBuffer(inQueueX, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(inQueueX2, 1, dataSize * sizeof(uint8_t));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
        AscendCUtils::SetOverflow(0);
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<srcType> srcLocal = inQueueX.AllocTensor<srcType>();
        LocalTensor<uint8_t> src1Local = inQueueX2.AllocTensor<uint8_t>();
        DataCopy(srcLocal, src_global, dataSize);
        inQueueX.EnQue(srcLocal);
        inQueueX2.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();

        LocalTensor<srcType> srcLocal = inQueueX.DeQue<srcType>();
        LocalTensor<uint8_t> tmpBuf = inQueueX2.DeQue<uint8_t>();
        srcType zero = 0;
        Duplicate(dstLocal, zero, dataSize);
        uint32_t count = (uint32_t)calCount;

        constexpr AscendC::CosAlgo algo = GetCosAlgo<algoMode>::value;
        static constexpr AscendC::CosConfig config = { algo };

        if constexpr (apiMode == 0) {
            Cos<srcType, false, config>(dstLocal, srcLocal);
        } else if constexpr (apiMode == 1) {
            Cos<srcType, false, config>(dstLocal, srcLocal, count);
        } else if constexpr (apiMode == 2) {
            Cos<srcType, false, config>(dstLocal, srcLocal, tmpBuf, count);
        } else {
            Cos<srcType, false, config>(dstLocal, srcLocal, tmpBuf);
        }

        outQueue.EnQue<srcType>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
        inQueueX2.FreeTensor(tmpBuf);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        DataCopy(dst_global, dstLocal, dataSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<srcType> src_global;
    GlobalTensor<srcType> dst_global;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueX2;
    TQue<QuePosition::VECOUT, 1> outQueue;
};

template <typename srcType, uint32_t calCount, uint32_t dataSize, uint32_t apiMode, uint32_t algoMode = 0>
__aicore__ void kernel_cos_operator(GM_ADDR src_gm, GM_ADDR dst_gm)
{
    KernelCos<srcType, calCount, dataSize, apiMode, algoMode> op;
    op.Init(src_gm, dst_gm);
    op.Process();
}

struct CosTestParams {
    uint32_t dataTypeSize;
    uint32_t inDataSize;
    void (*calFunc)(GM_ADDR, GM_ADDR);
};

class AdvanceCosTestSuite : public testing::TestWithParam<CosTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ADVANCE_API_COS, AdvanceCosTestSuite,
    ::testing::Values(
        CosTestParams {4, 32, kernel_cos_operator<float, 32, 32, 0, 0>},
        CosTestParams {4, 4096, kernel_cos_operator<float, 32, 4096, 1, 0>},
        CosTestParams {4, 2048, kernel_cos_operator<float, 2048, 2048, 2, 0>},
        CosTestParams {4, 1024, kernel_cos_operator<float, 128, 1024, 3, 0>},
        CosTestParams {4, 32, kernel_cos_operator<float, 32, 32, 0, 1>},
        CosTestParams {4, 4096, kernel_cos_operator<float, 32, 4096, 1, 1>},
        CosTestParams {4, 2048, kernel_cos_operator<float, 2048, 2048, 2, 1>},
        CosTestParams {4, 1024, kernel_cos_operator<float, 128, 1024, 3, 1>},
        CosTestParams {2, 32, kernel_cos_operator<half, 32, 32, 0, 0>},
        CosTestParams {2, 4096, kernel_cos_operator<half, 32, 4096, 1, 0>},
        CosTestParams {2, 2048, kernel_cos_operator<half, 2048, 2048, 2, 0>},
        CosTestParams {2, 1024, kernel_cos_operator<half, 128, 1024, 3, 0>},
        CosTestParams {2, 32, kernel_cos_operator<half, 32, 32, 0, 1>},
        CosTestParams {2, 4096, kernel_cos_operator<half, 32, 4096, 1, 1>},
        CosTestParams {2, 2048, kernel_cos_operator<half, 2048, 2048, 2, 1>},
        CosTestParams {2, 1024, kernel_cos_operator<half, 128, 1024, 3, 1>}
    )
);

TEST_P(AdvanceCosTestSuite, kernel_cos_operator)
{
    auto param = GetParam();
    uint32_t dataTypeSize = param.dataTypeSize;
    uint32_t inDataSize = param.inDataSize;
    uint8_t inputGm[inDataSize * dataTypeSize] = {1};
    uint8_t outputGm[inDataSize * dataTypeSize] = {0};
    param.calFunc(outputGm, inputGm);
    for (int32_t i = 0; i < inDataSize; i++) {
        EXPECT_EQ(outputGm[i], 0x00);
    }
}
