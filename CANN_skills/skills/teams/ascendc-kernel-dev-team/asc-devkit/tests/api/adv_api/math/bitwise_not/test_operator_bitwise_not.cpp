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
class KernelBitwiseNot {
public:
    __aicore__ inline KernelBitwiseNot()
    {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t srcSize)
    {
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), srcSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), srcSize);

        pipe.InitBuffer(inQueue, 1, srcSize * sizeof(T));
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
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, dataSize);
        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        BitwiseNot(dstLocal, srcLocal, this->dataSize);
        outQueue.EnQue<T>(dstLocal);
        inQueue.FreeTensor(srcLocal);
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
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t dataSize = 0;
};
}


template <typename T>
__aicore__ void testBitwiseNot(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t srcSize)
{
    KernelBitwiseNot<T> op;
    op.Init(dstGm, srcGm, srcSize);
    op.Process();
}


struct BitwiseNotTestParams {
    uint32_t dataTypeSize;
    uint32_t inDataSize;
    void (*calFunc)(GM_ADDR, GM_ADDR, uint32_t);
};


class BitwiseNotTestSuite : public testing::Test, public testing::WithParamInterface<BitwiseNotTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ADVANCE_API_BITWISE_NOT, BitwiseNotTestSuite,
    ::testing::Values(
        BitwiseNotTestParams {1, 1024, testBitwiseNot<uint8_t> },
        BitwiseNotTestParams {1, 32, testBitwiseNot<uint8_t> },
        BitwiseNotTestParams {1, 256, testBitwiseNot<uint8_t> },
        BitwiseNotTestParams {1, 1024, testBitwiseNot<int8_t> },
        BitwiseNotTestParams {1, 32, testBitwiseNot<int8_t> },
        BitwiseNotTestParams {1, 256, testBitwiseNot<int8_t> },
        BitwiseNotTestParams {2, 1024, testBitwiseNot<uint16_t> },
        BitwiseNotTestParams {2, 32, testBitwiseNot<uint16_t> },
        BitwiseNotTestParams {2, 256, testBitwiseNot<uint16_t> },
        BitwiseNotTestParams {2, 1024, testBitwiseNot<int16_t> },
        BitwiseNotTestParams {2, 32, testBitwiseNot<int16_t> },
        BitwiseNotTestParams {2, 256, testBitwiseNot<int16_t> },
        BitwiseNotTestParams {4, 1024, testBitwiseNot<uint32_t> },
        BitwiseNotTestParams {4, 32, testBitwiseNot<uint32_t> },
        BitwiseNotTestParams {4, 256, testBitwiseNot<uint32_t> },
        BitwiseNotTestParams {4, 1024, testBitwiseNot<int32_t> },
        BitwiseNotTestParams {4, 32, testBitwiseNot<int32_t> },
        BitwiseNotTestParams {4, 256, testBitwiseNot<int32_t> },
        BitwiseNotTestParams {8, 1024, testBitwiseNot<uint64_t> },
        BitwiseNotTestParams {8, 32, testBitwiseNot<uint64_t> },
        BitwiseNotTestParams {8, 256, testBitwiseNot<uint64_t> },
        BitwiseNotTestParams {8, 1024, testBitwiseNot<int64_t> },
        BitwiseNotTestParams {8, 32, testBitwiseNot<int64_t> },
        BitwiseNotTestParams {8, 256, testBitwiseNot<int64_t> }
    )
);

TEST_P(BitwiseNotTestSuite, testBitwiseNot)
{
    auto param = GetParam();
    uint32_t dataTypeSize = param.dataTypeSize;
    uint32_t inDataSize = param.inDataSize;
    uint8_t inputGm[inDataSize * dataTypeSize] = {1};
    uint8_t outputGm[inDataSize * dataTypeSize] = {0};
    param.calFunc(outputGm, inputGm, inDataSize);
    for (int32_t i = 0; i < inDataSize; i++) {
        EXPECT_EQ(outputGm[i], 0x00);
    }
}
