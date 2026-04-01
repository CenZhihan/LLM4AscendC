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
class KernelReduceXorSum {
public:
    __aicore__ inline KernelReduceXorSum() {}
    __aicore__ inline void Init(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t srcSize)
    {
        src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src0Gm), srcSize);
        src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src1Gm), srcSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dstGm), 16);

        pipe.InitBuffer(in0Queue, 1, srcSize * sizeof(srcType));
        pipe.InitBuffer(in1Queue, 1, srcSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, 16 * sizeof(srcType));
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
        LocalTensor<srcType> src0Local = in0Queue.AllocTensor<srcType>();
        DataCopy(src0Local, src0Global, bufferSize);
        in0Queue.EnQue(src0Local);

        LocalTensor<srcType> src1Local = in1Queue.AllocTensor<srcType>();
        DataCopy(src1Local, src1Global, bufferSize);
        in1Queue.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        LocalTensor<srcType> src0Local = in0Queue.DeQue<srcType>();
        LocalTensor<srcType> src1Local = in1Queue.DeQue<srcType>();
        ReduceXorSum<srcType, false>(dstLocal, src0Local, src1Local, bufferSize);
        outQueue.EnQue<srcType>(dstLocal);
        in0Queue.FreeTensor(src0Local);
        in1Queue.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        DataCopy(dstGlobal, dstLocal, 16);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<srcType> src0Global;
    GlobalTensor<srcType> src1Global;
    GlobalTensor<srcType> dstGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> in0Queue;
    TQue<TPosition::VECIN, 1> in1Queue;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t bufferSize = 0;
};

template <typename dataType>
__aicore__ void kernelReduceXorSumOperator(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t srcSize)
{
    KernelReduceXorSum<dataType> op;
    op.Init(src0Gm, src1Gm, dstGm, srcSize);
    op.Process();
}

class TEST_ReduceXorSum : public testing::Test {
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

#define VEC_ReduceXorSum_TESTCASE(DATA_TYPE, SRC_SIZE)                                  \
    TEST_F(TEST_ReduceXorSum, ReduceXorSum##DATA_TYPE##SRC_SIZE##Case)                  \
    {                                                                                   \
        uint8_t input0Gm[(SRC_SIZE) * sizeof(DATA_TYPE)];                                 \
        uint8_t input1Gm[(SRC_SIZE) * sizeof(DATA_TYPE)];                                 \
        uint8_t outputGm[16 * sizeof(DATA_TYPE)];                                       \
        for (uint32_t i = 0; i < (SRC_SIZE); i++) {                                       \
            input0Gm[i] = 0;                                                            \
            input1Gm[i] = 1;                                                            \
        }                                                                               \
        kernelReduceXorSumOperator<DATA_TYPE>(input0Gm, input1Gm, outputGm, SRC_SIZE);  \
                                                                                        \
        EXPECT_EQ(outputGm[0], 0);                                                      \
    }

VEC_ReduceXorSum_TESTCASE(int16_t, 1024);
VEC_ReduceXorSum_TESTCASE(int16_t, 16);
VEC_ReduceXorSum_TESTCASE(int16_t, 128);