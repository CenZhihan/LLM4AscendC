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
class KernelLogicalNot {
public:
    __aicore__ inline KernelLogicalNot()
    {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t srcSize)
    {
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(srcGm), srcSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), srcSize);

        pipe.InitBuffer(inQueue, 1, srcSize * sizeof(U));
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
        LocalTensor<U> srcLocal = inQueue.AllocTensor<U>();
        DataCopy(srcLocal, srcGlobal, dataSize);
        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<U> srcLocal = inQueue.DeQue<U>();
        LogicalNot(dstLocal, srcLocal, this->dataSize);
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
    GlobalTensor<U> srcGlobal;
    GlobalTensor<T> dstGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t dataSize = 0;
};
}


template <typename T, typename U>
__aicore__ void testLogicalNot(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t srcSize)
{
    KernelLogicalNot<T, U> op;
    op.Init(dstGm, srcGm, srcSize);
    op.Process();
}


struct LogicalNotTestParams {
    uint32_t dataTypeSize;
    uint32_t inDataSize;
    void (*calFunc)(GM_ADDR, GM_ADDR, uint32_t);
};


class LogicalNotTestSuite : public testing::Test, public testing::WithParamInterface<LogicalNotTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ADVANCE_API_LOGICAL_OR, LogicalNotTestSuite,
    ::testing::Values(
        LogicalNotTestParams {1, 1024, testLogicalNot<bool, bool> },
        LogicalNotTestParams {1, 32, testLogicalNot<bool, bool> },
        LogicalNotTestParams {1, 256, testLogicalNot<bool, bool> },
        LogicalNotTestParams {1, 1024, testLogicalNot<bool, uint8_t> },
        LogicalNotTestParams {1, 32, testLogicalNot<bool, uint8_t> },
        LogicalNotTestParams {1, 256, testLogicalNot<bool, uint8_t> },
        LogicalNotTestParams {1, 1024, testLogicalNot<bool, int8_t> },
        LogicalNotTestParams {1, 32, testLogicalNot<bool, int8_t> },
        LogicalNotTestParams {1, 256, testLogicalNot<bool, int8_t> },
        LogicalNotTestParams {2, 1024, testLogicalNot<bool, half> },
        LogicalNotTestParams {2, 32, testLogicalNot<bool, half> },
        LogicalNotTestParams {2, 256, testLogicalNot<bool, half> },
        LogicalNotTestParams {2, 1024, testLogicalNot<bool, bfloat16_t> },
        LogicalNotTestParams {2, 32, testLogicalNot<bool, bfloat16_t> },
        LogicalNotTestParams {2, 256, testLogicalNot<bool, bfloat16_t> },
        LogicalNotTestParams {2, 1024, testLogicalNot<bool, uint16_t> },
        LogicalNotTestParams {2, 32, testLogicalNot<bool, uint16_t> },
        LogicalNotTestParams {2, 256, testLogicalNot<bool, uint16_t> },
        LogicalNotTestParams {2, 1024, testLogicalNot<bool, int16_t> },
        LogicalNotTestParams {2, 32, testLogicalNot<bool, int16_t> },
        LogicalNotTestParams {2, 256, testLogicalNot<bool, int16_t> },
        LogicalNotTestParams {4, 1024, testLogicalNot<bool, float> },
        LogicalNotTestParams {4, 32, testLogicalNot<bool, float> },
        LogicalNotTestParams {4, 256, testLogicalNot<bool, float> },
        LogicalNotTestParams {4, 1024, testLogicalNot<bool, uint32_t> },
        LogicalNotTestParams {4, 32, testLogicalNot<bool, uint32_t> },
        LogicalNotTestParams {4, 256, testLogicalNot<bool, uint32_t> },
        LogicalNotTestParams {4, 1024, testLogicalNot<bool, int32_t> },
        LogicalNotTestParams {4, 32, testLogicalNot<bool, int32_t> },
        LogicalNotTestParams {4, 256, testLogicalNot<bool, int32_t> },
        LogicalNotTestParams {8, 1024, testLogicalNot<bool, uint64_t> },
        LogicalNotTestParams {8, 32, testLogicalNot<bool, uint64_t> },
        LogicalNotTestParams {8, 256, testLogicalNot<bool, uint64_t> },
        LogicalNotTestParams {8, 1024, testLogicalNot<bool, int64_t> },
        LogicalNotTestParams {8, 32, testLogicalNot<bool, int64_t> },
        LogicalNotTestParams {8, 256, testLogicalNot<bool, int64_t> }
    )
);

TEST_P(LogicalNotTestSuite, testLogicalNot)
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