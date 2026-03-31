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
#include <type_traits>
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;

template <typename T, typename U, int16_t vF>
class KernelVecBinary {
public:
    __aicore__ inline KernelVecBinary()
    {}
    __aicore__ inline void Init(GM_ADDR srcGm0, GM_ADDR srcGm1, GM_ADDR dstGm0, GM_ADDR dstGm1, uint32_t dataSizeIn, uint32_t calCountIn)
    {
        calCount = calCountIn;
        const int srcAlignSize = 32 / sizeof(T);
        srcDataSize = CeilDivision(dataSizeIn, srcAlignSize) * srcAlignSize;
        const int dstAlignSize = 32 / sizeof(U);
        dstDataSize = CeilDivision(dataSizeIn, dstAlignSize) * dstAlignSize;
        srcGlobalX.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm0), srcDataSize);
        srcGlobalY.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm1), srcDataSize);
        dstGlobalX.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(dstGm0), dstDataSize);
        dstGlobalY.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(dstGm1), dstDataSize);
        pipe.InitBuffer(inQueueX, 1, srcDataSize * sizeof(T));
        pipe.InitBuffer(inQueueY, 1, srcDataSize * sizeof(T));
        pipe.InitBuffer(outQueueX, 1, dstDataSize * sizeof(U));
        pipe.InitBuffer(outQueueY, 1, dstDataSize * sizeof(U));
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
        LocalTensor<T> srcLocalX = inQueueX.AllocTensor<T>();
        DataCopy(srcLocalX, srcGlobalX, srcDataSize);
        inQueueX.EnQue<T>(srcLocalX);

        LocalTensor<T> srcLocalY = inQueueY.AllocTensor<T>();
        DataCopy(srcLocalY, srcGlobalY, srcDataSize);
        inQueueY.EnQue<T>(srcLocalY);

    }

    __aicore__ inline void Compute()
    {
        uint32_t repeatElm = GetVecLen() / sizeof(T);
        uint32_t repeatTimes = srcDataSize / repeatElm;
        LocalTensor<U> dstLocalX = outQueueX.AllocTensor<U>();
        LocalTensor<U> dstLocalY = outQueueY.AllocTensor<U>();
        LocalTensor<T> srcLocalX = inQueueX.DeQue<T>();
        LocalTensor<T> srcLocalY = inQueueY.DeQue<T>();
        const U Zero = 0;
        Duplicate(dstLocalX, Zero, dstDataSize);
        Duplicate(dstLocalY, Zero, dstDataSize);
     
        #ifdef vF
        #if vF == 0
            Neg(dstLocalX, srcLocalY, calCount);
        #endif
        #endif
        outQueueX.EnQue<U>(dstLocalX);
        outQueueY.EnQue<U>(dstLocalY);
        inQueueX.FreeTensor(srcLocalX);
        inQueueY.FreeTensor(srcLocalY);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<U> dstLocalX = outQueueX.DeQue<U>();
        LocalTensor<U> dstLocalY = outQueueY.DeQue<U>();
        
        DataCopy(dstGlobalX, dstLocalX, dstDataSize);
        DataCopy(dstGlobalY, dstLocalY, dstDataSize);
        outQueueX.FreeTensor(dstLocalX);
        outQueueY.FreeTensor(dstLocalY);
    }

private:
    GlobalTensor<T> srcGlobalX;
    GlobalTensor<T> srcGlobalY;
    GlobalTensor<U> dstGlobalX;
    GlobalTensor<U> dstGlobalY;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECIN, 1> inQueueY;
    TQue<TPosition::VECOUT, 1> outQueueX;
    TQue<TPosition::VECOUT, 1> outQueueY;
    uint32_t calCount;
    uint32_t srcDataSize;
    uint32_t dstDataSize;
};
 
struct MicroVecUnaryParams {
    void (*CallFunc)();
};
 
template<typename T, typename U, int16_t vF>
void RunCase() {
    int srcByteSize = sizeof(T);
    int dstByteSize = sizeof(U);
    int dataSize = 256;
    int mask = 256;
    uint8_t dstGm0[dataSize * dstByteSize] = {0};
    uint8_t dstGm1[dataSize * dstByteSize] = {0};
    uint8_t srcGm0[dataSize * srcByteSize] = {0};
    uint8_t srcGm1[dataSize * srcByteSize] = {0};
 
    KernelVecBinary<T, U, vF> op;
    op.Init(srcGm0, srcGm1, dstGm0, dstGm1, dataSize, mask);
    op.Process();
}

class MicroVecUnaryTestSuite : public testing::Test, public testing::WithParamInterface<MicroVecUnaryParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};
 
INSTANTIATE_TEST_CASE_P(MicroVecUnaryTestCases, MicroVecUnaryTestSuite,
    ::testing::Values(MicroVecUnaryParams { RunCase<float, float, 0>},
                      MicroVecUnaryParams { RunCase<half, half, 0>},
                      MicroVecUnaryParams { RunCase<int8_t, int8_t, 0>},
                      MicroVecUnaryParams { RunCase<int16_t, int16_t, 0>},
                      MicroVecUnaryParams { RunCase<int32_t, int32_t, 0>}
                      ));
 
TEST_P(MicroVecUnaryTestSuite, MicroVecUnaryTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}