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

namespace AscendC {

template <int size = 0> struct GetPowerAlgo {
    static constexpr AscendC::PowerAlgo value = AscendC::PowerAlgo::INTRINSIC;
};

template <> struct GetPowerAlgo<1> {
    static constexpr AscendC::PowerAlgo value = AscendC::PowerAlgo::DOUBLE_FLOAT_TECH;
};

template <typename srcType, uint32_t algoMode = 0>
class KernelPower {
public:
    __aicore__ inline KernelPower()
    {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR srcGmBase, GM_ADDR srcGmExp, uint32_t srcSize, uint32_t mode)
    {
        srcGlobalBase.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(srcGmBase), srcSize);
        srcGlobalExp.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(srcGmExp), srcSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dstGm), srcSize);

        pipe.InitBuffer(inQueueX1, 1, srcSize * sizeof(srcType));
        pipe.InitBuffer(inQueueX2, 1, srcSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, srcSize * sizeof(srcType));
        this->bufferSize = srcSize;
        this->mode = mode;
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
        AscendC::AscendCUtils::SetOverflow(0);
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<srcType> srcLocalBase = inQueueX1.AllocTensor<srcType>();
        DataCopy(srcLocalBase, srcGlobalBase, bufferSize);
        inQueueX1.EnQue(srcLocalBase);

        LocalTensor<srcType> srcLocalExp = inQueueX2.AllocTensor<srcType>();
        DataCopy(srcLocalExp, srcGlobalExp, bufferSize);
        inQueueX2.EnQue(srcLocalExp);
    }
    __aicore__ inline void Compute()
    {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        LocalTensor<srcType> srcLocalBase = inQueueX1.DeQue<srcType>();
        LocalTensor<srcType> srcLocalExp = inQueueX2.DeQue<srcType>();

        constexpr AscendC::PowerAlgo algo = GetPowerAlgo<algoMode>::value;
        static constexpr AscendC::PowerConfig config = { algo };
        if (mode == 0) {
            Power<srcType, false, config>(dstLocal, srcLocalBase, srcLocalExp);
        } else if (mode == 1) {
            srcType scalarValue = srcLocalExp.GetValue(0);
            PipeBarrier<PIPE_ALL>();
            Power<srcType, false, config>(dstLocal, srcLocalBase, scalarValue);
        } else if (mode == 2) {
            srcType scalarValue = srcLocalBase.GetValue(0);
            PipeBarrier<PIPE_ALL>();
            Power<srcType, false, config>(dstLocal, scalarValue, srcLocalExp);
        }

        outQueue.EnQue<srcType>(dstLocal);
        inQueueX1.FreeTensor(srcLocalBase);
        inQueueX2.FreeTensor(srcLocalExp);

    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        int32_t remainVal = bufferSize % (32 / sizeof(srcType));
        if (remainVal != bufferSize) {
            DataCopy(dstGlobal, dstLocal, bufferSize - remainVal);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t i = bufferSize - remainVal; i < bufferSize; i++) {
            dstGlobal.SetValue(i, dstLocal.GetValue(i));
        }
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<srcType> srcGlobalBase;
    GlobalTensor<srcType> srcGlobalExp;
    GlobalTensor<srcType> dstGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX1;
    TQue<TPosition::VECIN, 1> inQueueX2;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t bufferSize = 0;
    uint32_t mode = 0;
};
}

template <typename T, uint32_t algoMode>
__aicore__ void testPower(GM_ADDR dstGm, GM_ADDR srcGmBase, GM_ADDR srcGmExp, uint32_t srcSize, uint32_t mode)
{
    AscendC::KernelPower<T, algoMode> op;
    op.Init(dstGm, srcGmBase, srcGmExp, srcSize, mode);
    op.Process();
}


struct PowerTestParams {
    uint32_t dataTypeSize;
    uint32_t inDataSize;
    uint32_t mode;
    void (*calFunc)(GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t);
};


class AdvancePowerTestSuite : public testing::Test, public testing::WithParamInterface<PowerTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ADVANCE_API_POWER, AdvancePowerTestSuite,
    ::testing::Values(
        PowerTestParams {4, 32, 0, testPower<float, 1>},
        PowerTestParams {4, 4096, 1, testPower<float, 1> },
        PowerTestParams {4, 512, 2, testPower<float, 1> },
        PowerTestParams {2, 16, 0, testPower<half, 1> },
        PowerTestParams {2, 1024, 1, testPower<half, 1> },
        PowerTestParams {2, 256, 2, testPower<half, 1> },
        PowerTestParams {2, 32, 0, testPower<bfloat16_t, 1> },
        PowerTestParams {2, 64, 1, testPower<bfloat16_t, 1> },
        PowerTestParams {2, 128, 2, testPower<bfloat16_t, 1> },
        PowerTestParams {1, 1024, 0, testPower<uint8_t, 0> },
        PowerTestParams {1, 32, 1, testPower<uint8_t, 0> },
        PowerTestParams {1, 256, 2, testPower<uint8_t, 0> },
        PowerTestParams {1, 1024, 0, testPower<int8_t, 0> },
        PowerTestParams {1, 32, 1, testPower<int8_t, 0> },
        PowerTestParams {1, 256, 2, testPower<int8_t, 0> },
        PowerTestParams {2, 1024, 0, testPower<uint16_t, 0> },
        PowerTestParams {2, 32, 1, testPower<uint16_t, 0> },
        PowerTestParams {2, 256, 2, testPower<uint16_t, 0> },
        PowerTestParams {2, 1024, 0, testPower<int16_t, 0> },
        PowerTestParams {2, 32, 1, testPower<int16_t, 0> },
        PowerTestParams {2, 256, 2, testPower<int16_t, 0> },
        PowerTestParams {4, 1024, 0, testPower<uint32_t, 0> },
        PowerTestParams {4, 32, 1, testPower<uint32_t, 0> },
        PowerTestParams {4, 256, 2, testPower<uint32_t, 0> },
        PowerTestParams {4, 1024, 0, testPower<int32_t, 0> },
        PowerTestParams {4, 32, 1, testPower<int32_t, 0> },
        PowerTestParams {4, 256, 2, testPower<int32_t, 0> }
    )
);

TEST_P(AdvancePowerTestSuite, testPower)
{
    auto param = GetParam();
    uint32_t dataTypeSize = param.dataTypeSize;
    uint32_t inDataSize = param.inDataSize;
    uint8_t inputGmBase[inDataSize * dataTypeSize] = {1};
    uint8_t inputGmExp[inDataSize * dataTypeSize] = {1};
    uint8_t outputGm[inDataSize * dataTypeSize] = {0};
    param.calFunc(outputGm, inputGmBase, inputGmExp, inDataSize, param.mode);
    for (int32_t i = 0; i < inDataSize; i++) {
        EXPECT_EQ(outputGm[i], 0x00);
    }
}
