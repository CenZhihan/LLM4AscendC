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
enum class ScalarType : std::uint8_t{
    TENSOR_SCALAR,
    SCALAR_TENSOR,
    TENSOR_INDEX0,
    TENSOR_INDEX1
};
template <typename T, typename U, ScalarType scalarType>
class KernelLogicalOrs {
public:
    __aicore__ inline KernelLogicalOrs()
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
        U scalar = static_cast<U>(3);
        if constexpr (scalarType == ScalarType::SCALAR_TENSOR) {
            LogicalOrs(dstLocal, src0Local, scalar, dataSize);
        } else if constexpr (scalarType == ScalarType::TENSOR_SCALAR) {
            LogicalOrs(dstLocal, scalar, src1Local, dataSize);
        } else if constexpr (scalarType == ScalarType::TENSOR_INDEX0) {
            static constexpr LogicalOrsConfig config = {false, 0};
            LogicalOrs<config>(dstLocal, src0Local, src1Local, dataSize);
        } else if constexpr (scalarType == ScalarType::TENSOR_INDEX1) {
            LogicalOrs(dstLocal, src0Local, src1Local, dataSize);
        } 
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


template <typename T, typename U, ScalarType scalarType>
__aicore__ void testLogicalOrs(GM_ADDR dstGm, GM_ADDR src0Gm, GM_ADDR src1Gm, uint32_t srcSize)
{
    KernelLogicalOrs<T, U, scalarType> op;
    op.Init(dstGm, src0Gm, src1Gm, srcSize);
    op.Process();
}


struct LogicalOrsTestParams {
    uint32_t dataTypeSize;
    uint32_t inDataSize;
    void (*calFunc)(GM_ADDR, GM_ADDR, GM_ADDR, uint32_t);
};


class LogicalOrsTestSuite : public testing::Test, public testing::WithParamInterface<LogicalOrsTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ADVANCE_API_LOGICAL_ORS, LogicalOrsTestSuite,
    ::testing::Values(
        LogicalOrsTestParams {1, 1024, testLogicalOrs<bool, bool, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {1, 32, testLogicalOrs<bool, bool, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {1, 256, testLogicalOrs<bool, bool, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {1, 1024, testLogicalOrs<bool, uint8_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {1, 32, testLogicalOrs<bool, uint8_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {1, 256, testLogicalOrs<bool, uint8_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {1, 1024, testLogicalOrs<bool, int8_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {1, 32, testLogicalOrs<bool, int8_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {1, 256, testLogicalOrs<bool, int8_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, half, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, half, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, half, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, bfloat16_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, bfloat16_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, bfloat16_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, uint16_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, uint16_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, uint16_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, int16_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, int16_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, int16_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {4, 1024, testLogicalOrs<bool, float, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {4, 32, testLogicalOrs<bool, float, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {4, 256, testLogicalOrs<bool, float, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {4, 1024, testLogicalOrs<bool, uint32_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {4, 32, testLogicalOrs<bool, uint32_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {4, 256, testLogicalOrs<bool, uint32_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {4, 1024, testLogicalOrs<bool, int32_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {4, 32, testLogicalOrs<bool, int32_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {4, 256, testLogicalOrs<bool, int32_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {8, 1024, testLogicalOrs<bool, uint64_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {8, 32, testLogicalOrs<bool, uint64_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {8, 256, testLogicalOrs<bool, uint64_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {8, 1024, testLogicalOrs<bool, int64_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {8, 32, testLogicalOrs<bool, int64_t, ScalarType::TENSOR_SCALAR> },
        LogicalOrsTestParams {8, 256, testLogicalOrs<bool, int64_t, ScalarType::TENSOR_SCALAR> },

        LogicalOrsTestParams {1, 1024, testLogicalOrs<bool, bool, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {1, 32, testLogicalOrs<bool, bool, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {1, 256, testLogicalOrs<bool, bool, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {1, 1024, testLogicalOrs<bool, uint8_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {1, 32, testLogicalOrs<bool, uint8_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {1, 256, testLogicalOrs<bool, uint8_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {1, 1024, testLogicalOrs<bool, int8_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {1, 32, testLogicalOrs<bool, int8_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {1, 256, testLogicalOrs<bool, int8_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, half, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, half, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, half, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, bfloat16_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, bfloat16_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, bfloat16_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, uint16_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, uint16_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, uint16_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, int16_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, int16_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, int16_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {4, 1024, testLogicalOrs<bool, float, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {4, 32, testLogicalOrs<bool, float, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {4, 256, testLogicalOrs<bool, float, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {4, 1024, testLogicalOrs<bool, uint32_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {4, 32, testLogicalOrs<bool, uint32_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {4, 256, testLogicalOrs<bool, uint32_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {4, 1024, testLogicalOrs<bool, int32_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {4, 32, testLogicalOrs<bool, int32_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {4, 256, testLogicalOrs<bool, int32_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {8, 1024, testLogicalOrs<bool, uint64_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {8, 32, testLogicalOrs<bool, uint64_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {8, 256, testLogicalOrs<bool, uint64_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {8, 1024, testLogicalOrs<bool, int64_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {8, 32, testLogicalOrs<bool, int64_t, ScalarType::SCALAR_TENSOR> },
        LogicalOrsTestParams {8, 256, testLogicalOrs<bool, int64_t, ScalarType::SCALAR_TENSOR> },

        LogicalOrsTestParams {1, 1024, testLogicalOrs<bool, bool, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {1, 32, testLogicalOrs<bool, bool, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {1, 256, testLogicalOrs<bool, bool, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {1, 1024, testLogicalOrs<bool, uint8_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {1, 32, testLogicalOrs<bool, uint8_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {1, 256, testLogicalOrs<bool, uint8_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {1, 1024, testLogicalOrs<bool, int8_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {1, 32, testLogicalOrs<bool, int8_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {1, 256, testLogicalOrs<bool, int8_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, half, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, half, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, half, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, bfloat16_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, bfloat16_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, bfloat16_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, uint16_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, uint16_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, uint16_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, int16_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, int16_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, int16_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {4, 1024, testLogicalOrs<bool, float, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {4, 32, testLogicalOrs<bool, float, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {4, 256, testLogicalOrs<bool, float, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {4, 1024, testLogicalOrs<bool, uint32_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {4, 32, testLogicalOrs<bool, uint32_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {4, 256, testLogicalOrs<bool, uint32_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {4, 1024, testLogicalOrs<bool, int32_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {4, 32, testLogicalOrs<bool, int32_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {4, 256, testLogicalOrs<bool, int32_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {8, 1024, testLogicalOrs<bool, uint64_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {8, 32, testLogicalOrs<bool, uint64_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {8, 256, testLogicalOrs<bool, uint64_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {8, 1024, testLogicalOrs<bool, int64_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {8, 32, testLogicalOrs<bool, int64_t, ScalarType::TENSOR_INDEX0> },
        LogicalOrsTestParams {8, 256, testLogicalOrs<bool, int64_t, ScalarType::TENSOR_INDEX0> },

        LogicalOrsTestParams {1, 1024, testLogicalOrs<bool, bool, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {1, 32, testLogicalOrs<bool, bool, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {1, 256, testLogicalOrs<bool, bool, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {1, 1024, testLogicalOrs<bool, uint8_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {1, 32, testLogicalOrs<bool, uint8_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {1, 256, testLogicalOrs<bool, uint8_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {1, 1024, testLogicalOrs<bool, int8_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {1, 32, testLogicalOrs<bool, int8_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {1, 256, testLogicalOrs<bool, int8_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, half, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, half, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, half, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, bfloat16_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, bfloat16_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, bfloat16_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, uint16_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, uint16_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, uint16_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {2, 1024, testLogicalOrs<bool, int16_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {2, 32, testLogicalOrs<bool, int16_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {2, 256, testLogicalOrs<bool, int16_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {4, 1024, testLogicalOrs<bool, float, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {4, 32, testLogicalOrs<bool, float, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {4, 256, testLogicalOrs<bool, float, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {4, 1024, testLogicalOrs<bool, uint32_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {4, 32, testLogicalOrs<bool, uint32_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {4, 256, testLogicalOrs<bool, uint32_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {4, 1024, testLogicalOrs<bool, int32_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {4, 32, testLogicalOrs<bool, int32_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {4, 256, testLogicalOrs<bool, int32_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {8, 1024, testLogicalOrs<bool, uint64_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {8, 32, testLogicalOrs<bool, uint64_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {8, 256, testLogicalOrs<bool, uint64_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {8, 1024, testLogicalOrs<bool, int64_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {8, 32, testLogicalOrs<bool, int64_t, ScalarType::TENSOR_INDEX1> },
        LogicalOrsTestParams {8, 256, testLogicalOrs<bool, int64_t, ScalarType::TENSOR_INDEX1> }
    )
);

TEST_P(LogicalOrsTestSuite, testLogicalOrs)
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

