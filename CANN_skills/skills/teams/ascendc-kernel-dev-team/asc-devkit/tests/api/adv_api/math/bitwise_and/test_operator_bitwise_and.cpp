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
using namespace std;
using namespace AscendC;

namespace AscendC {
template <typename T>
class KernelBitwiseAnd {
public:
    __aicore__ inline KernelBitwiseAnd()
    {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR src0Gm, GM_ADDR src1Gm, uint32_t srcSize)
    {
        src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src0Gm), srcSize);
        src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src1Gm), srcSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), srcSize);

        pipe.InitBuffer(inQueueX0, 1, srcSize * sizeof(T));
        pipe.InitBuffer(inQueueX1, 1, srcSize * sizeof(T));
        pipe.InitBuffer(outQueue, 1, srcSize * sizeof(T));
        this->dataSize = srcSize;
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
        LocalTensor<T> src0Local = inQueueX0.AllocTensor<T>();
        DataCopy(src0Local, src0Global, dataSize);
        inQueueX0.EnQue(src0Local);

        LocalTensor<T> src1Local = inQueueX1.AllocTensor<T>();
        DataCopy(src1Local, src1Global, dataSize);
        inQueueX1.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> src0Local = inQueueX0.DeQue<T>();
        LocalTensor<T> src1Local = inQueueX1.DeQue<T>();
        BitwiseAnd(dstLocal, src0Local, src1Local, this->dataSize);
        outQueue.EnQue<T>(dstLocal);
        inQueueX0.FreeTensor(src0Local);
        inQueueX1.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        int32_t remainVal = dataSize % (32 / sizeof(T));
        if (remainVal != dataSize) {
            DataCopy(dstGlobal, dstLocal, dataSize - remainVal);
        }
        PipeBarrier<PIPE_ALL>();
        for (int32_t i = dataSize - remainVal; i < dataSize; i++) {
            dstGlobal.SetValue(i, dstLocal.GetValue(i));
        }
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<T> src0Global;
    GlobalTensor<T> src1Global;
    GlobalTensor<T> dstGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX0;
    TQue<TPosition::VECIN, 1> inQueueX1;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t dataSize = 0;
};
}


template <typename T>
__aicore__ void testBitwiseAnd(GM_ADDR dstGm, GM_ADDR src0Gm, GM_ADDR src1Gm, uint32_t srcSize)
{
    KernelBitwiseAnd<T> op;
    op.Init(dstGm, src0Gm, src1Gm, srcSize);
    op.Process();
}


struct BitwiseAndTestParams {
    uint32_t dataTypeSize;
    uint32_t inDataSize;
    void (*calFunc)(GM_ADDR, GM_ADDR, GM_ADDR, uint32_t);
};


class BitwiseAndTestSuite : public testing::Test, public testing::WithParamInterface<BitwiseAndTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ADVANCE_API_BITWISE_AND, BitwiseAndTestSuite,
    ::testing::Values(
        BitwiseAndTestParams {1, 1024, testBitwiseAnd<uint8_t> },
        BitwiseAndTestParams {1, 32, testBitwiseAnd<uint8_t> },
        BitwiseAndTestParams {1, 256, testBitwiseAnd<uint8_t> },
        BitwiseAndTestParams {1, 1024, testBitwiseAnd<int8_t> },
        BitwiseAndTestParams {1, 32, testBitwiseAnd<int8_t> },
        BitwiseAndTestParams {1, 256, testBitwiseAnd<int8_t> },
        BitwiseAndTestParams {2, 1024, testBitwiseAnd<uint16_t> },
        BitwiseAndTestParams {2, 32, testBitwiseAnd<uint16_t> },
        BitwiseAndTestParams {2, 256, testBitwiseAnd<uint16_t> },
        BitwiseAndTestParams {2, 1024, testBitwiseAnd<int16_t> },
        BitwiseAndTestParams {2, 32, testBitwiseAnd<int16_t> },
        BitwiseAndTestParams {2, 256, testBitwiseAnd<int16_t> },
        BitwiseAndTestParams {4, 1024, testBitwiseAnd<uint32_t> },
        BitwiseAndTestParams {4, 32, testBitwiseAnd<uint32_t> },
        BitwiseAndTestParams {4, 256, testBitwiseAnd<uint32_t> },
        BitwiseAndTestParams {4, 1024, testBitwiseAnd<int32_t> },
        BitwiseAndTestParams {4, 32, testBitwiseAnd<int32_t> },
        BitwiseAndTestParams {4, 256, testBitwiseAnd<int32_t> },
        BitwiseAndTestParams {8, 1024, testBitwiseAnd<uint64_t> },
        BitwiseAndTestParams {8, 32, testBitwiseAnd<uint64_t> },
        BitwiseAndTestParams {8, 256, testBitwiseAnd<uint64_t> },
        BitwiseAndTestParams {8, 1024, testBitwiseAnd<int64_t> },
        BitwiseAndTestParams {8, 32, testBitwiseAnd<int64_t> },
        BitwiseAndTestParams {8, 256, testBitwiseAnd<int64_t> }
    )
);

TEST_P(BitwiseAndTestSuite, testBitwiseAnd)
{
    auto param = GetParam();
    uint32_t dataTypeSize = param.dataTypeSize;
    uint32_t inDataSize = param.inDataSize;
    uint8_t inputGm0[inDataSize * dataTypeSize] = {1};
    uint8_t inputGm1[inDataSize * dataTypeSize] = {1};
    uint8_t outputGm[inDataSize * dataTypeSize] = {0};
    param.calFunc(outputGm, inputGm0, inputGm1, inDataSize);
    for (int32_t i = 0; i < inDataSize; i++) {
        EXPECT_EQ(outputGm[i], 0x00);
    }
}
