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
template <typename T, typename U>
class KernelLogicalXor {
public:
    __aicore__ inline KernelLogicalXor()
    {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR src0Gm, GM_ADDR src1Gm, uint32_t srcSize)
    {
        src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(src0Gm), srcSize);
        src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(src1Gm), srcSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), srcSize);

        pipe.InitBuffer(inQueueX0, 1, srcSize * sizeof(U));
        pipe.InitBuffer(inQueueX1, 1, srcSize * sizeof(U));
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
        LocalTensor<U> src0Local = inQueueX0.AllocTensor<U>();
        DataCopy(src0Local, src0Global, dataSize);
        inQueueX0.EnQue(src0Local);

        LocalTensor<U> src1Local = inQueueX1.AllocTensor<U>();
        DataCopy(src1Local, src1Global, dataSize);
        inQueueX1.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<U> src0Local = inQueueX0.DeQue<U>();
        LocalTensor<U> src1Local = inQueueX1.DeQue<U>();
        LogicalXor(dstLocal, src0Local, src1Local, this->dataSize);
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
    GlobalTensor<U> src0Global;
    GlobalTensor<U> src1Global;
    GlobalTensor<T> dstGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX0;
    TQue<TPosition::VECIN, 1> inQueueX1;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t dataSize = 0;
};
}


template <typename T, typename U>
__aicore__ void testLogicalXor(GM_ADDR dstGm, GM_ADDR src0Gm, GM_ADDR src1Gm, uint32_t srcSize)
{
    KernelLogicalXor<T, U> op;
    op.Init(dstGm, src0Gm, src1Gm, srcSize);
    op.Process();
}


struct LogicalXorTestParams {
    uint32_t dataTypeSize;
    uint32_t inDataSize;
    void (*calFunc)(GM_ADDR, GM_ADDR, GM_ADDR, uint32_t);
};


class LogicalXorTestSuite : public testing::Test, public testing::WithParamInterface<LogicalXorTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ADVANCE_API_LOGICAL_XOR, LogicalXorTestSuite,
    ::testing::Values(
        LogicalXorTestParams {1, 1024, testLogicalXor<bool, bool> },
        LogicalXorTestParams {1, 32, testLogicalXor<bool, bool> },
        LogicalXorTestParams {1, 256, testLogicalXor<bool, bool> },
        LogicalXorTestParams {1, 1024, testLogicalXor<bool, uint8_t> },
        LogicalXorTestParams {1, 32, testLogicalXor<bool, uint8_t> },
        LogicalXorTestParams {1, 256, testLogicalXor<bool, uint8_t> },
        LogicalXorTestParams {1, 1024, testLogicalXor<bool, int8_t> },
        LogicalXorTestParams {1, 32, testLogicalXor<bool, int8_t> },
        LogicalXorTestParams {1, 256, testLogicalXor<bool, int8_t> },
        LogicalXorTestParams {2, 1024, testLogicalXor<bool, half> },
        LogicalXorTestParams {2, 32, testLogicalXor<bool, half> },
        LogicalXorTestParams {2, 256, testLogicalXor<bool, half> },
        LogicalXorTestParams {2, 1024, testLogicalXor<bool, bfloat16_t> },
        LogicalXorTestParams {2, 32, testLogicalXor<bool, bfloat16_t> },
        LogicalXorTestParams {2, 256, testLogicalXor<bool, bfloat16_t> },
        LogicalXorTestParams {2, 1024, testLogicalXor<bool, uint16_t> },
        LogicalXorTestParams {2, 32, testLogicalXor<bool, uint16_t> },
        LogicalXorTestParams {2, 256, testLogicalXor<bool, uint16_t> },
        LogicalXorTestParams {2, 1024, testLogicalXor<bool, int16_t> },
        LogicalXorTestParams {2, 32, testLogicalXor<bool, int16_t> },
        LogicalXorTestParams {2, 256, testLogicalXor<bool, int16_t> },
        LogicalXorTestParams {4, 1024, testLogicalXor<bool, float> },
        LogicalXorTestParams {4, 32, testLogicalXor<bool, float> },
        LogicalXorTestParams {4, 256, testLogicalXor<bool, float> },
        LogicalXorTestParams {4, 1024, testLogicalXor<bool, uint32_t> },
        LogicalXorTestParams {4, 32, testLogicalXor<bool, uint32_t> },
        LogicalXorTestParams {4, 256, testLogicalXor<bool, uint32_t> },
        LogicalXorTestParams {4, 1024, testLogicalXor<bool, int32_t> },
        LogicalXorTestParams {4, 32, testLogicalXor<bool, int32_t> },
        LogicalXorTestParams {4, 256, testLogicalXor<bool, int32_t> },
        LogicalXorTestParams {8, 1024, testLogicalXor<bool, uint64_t> },
        LogicalXorTestParams {8, 32, testLogicalXor<bool, uint64_t> },
        LogicalXorTestParams {8, 256, testLogicalXor<bool, uint64_t> },
        LogicalXorTestParams {8, 1024, testLogicalXor<bool, int64_t> },
        LogicalXorTestParams {8, 32, testLogicalXor<bool, int64_t> },
        LogicalXorTestParams {8, 256, testLogicalXor<bool, int64_t> }
    )
);

TEST_P(LogicalXorTestSuite, testLogicalXor)
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