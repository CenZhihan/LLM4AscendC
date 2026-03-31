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

template <typename srcType>
class KernelDigamma {
public:
    __aicore__ inline KernelDigamma() {}
    __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t srcSize)
    {
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(srcGm), srcSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dstGm), srcSize);

        pipe.InitBuffer(inQueue, 1, srcSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, srcSize * sizeof(srcType));
        bufferSize = srcSize;
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
        LocalTensor<srcType> srcLocal = inQueue.AllocTensor<srcType>();
        DataCopy(srcLocal, srcGlobal, bufferSize);
        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        LocalTensor<srcType> srcLocal = inQueue.DeQue<srcType>();
        Digamma<srcType, false>(dstLocal, srcLocal, bufferSize);
        outQueue.EnQue<srcType>(dstLocal);
        inQueue.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        DataCopy(dstGlobal, dstLocal, bufferSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<srcType> srcGlobal;
    GlobalTensor<srcType> dstGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t bufferSize = 0;
};

template <typename dataType>
__aicore__ void kernelDigammaOperator(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t srcSize)
{
    KernelDigamma<dataType> op;
    op.Init(srcGm, dstGm, srcSize);
    op.Process();
}

class TEST_DIGAMMA : public testing::Test {
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

#define VEC_DIGAMMA_TESTCASE(DATA_TYPE)                                              \
    TEST_F(TEST_DIGAMMA, Digamma##DATA_TYPE##Case)                                   \
    {                                                                                \
        uint32_t dataSize = 256;                                                     \
        uint32_t selMaskSize = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t)); \
        uint8_t inputGm[dataSize * sizeof(DATA_TYPE)];                               \
        uint8_t outputGm[dataSize * sizeof(DATA_TYPE)];                              \
        kernelDigammaOperator<DATA_TYPE>(inputGm, outputGm, dataSize);               \
                                                                                     \
        for (uint32_t i = 0; i < dataSize; i++) {                                    \
            EXPECT_EQ(outputGm[i], 0x00);                                            \
        }                                                                            \
    }

VEC_DIGAMMA_TESTCASE(float);
VEC_DIGAMMA_TESTCASE(half);