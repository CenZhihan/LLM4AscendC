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

constexpr DivConfig config = { DivAlgo::DIFF_COMPENSATION };

template <typename T, typename U, int16_t VF>
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
     
        #ifdef VF
        #if VF == 0
            Div<T, config>(dstLocalX, srcLocalX, srcLocalY, calCount);
        #elif VF == 1
            Prelu(dstLocalX, srcLocalX, srcLocalY, calCount);
        #elif VF == 2
            Mull(dstLocalX, dstLocalY, srcLocalX, srcLocalY, calCount);
        #elif VF == 3
            FusedAbsSub(dstLocalX, srcLocalX, srcLocalY, calCount);
        #elif VF == 4
            FusedExpSub(dstLocalX, srcLocalX, srcLocalY, calCount);
        #elif VF == 5
            T scalar = srcLocalX.GetValue(0);
            FusedMulsCast(dstLocalX, scalar, srcLocalY, calCount);
        #elif VF == 6
            T scalar = srcLocalY.GetValue(0);
            FusedMulsCast(dstLocalX, srcLocalX, scalar, calCount);
        #elif VF == 7
            AddRelu(dstLocalX, srcLocalX, srcLocalY, calCount);
        #elif VF == 8
            AddReluCast(dstLocalX, srcLocalX, srcLocalY, calCount);
        #elif VF == 9
            SubRelu(dstLocalX, srcLocalX, srcLocalY, calCount);
        #elif VF == 10
            SubReluCast(dstLocalX, srcLocalX, srcLocalY, calCount);
        #elif VF == 11
            FusedMulAdd(dstLocalX, srcLocalX, srcLocalY, calCount);
        #elif VF == 12
            MulCast(dstLocalX, srcLocalX, srcLocalY, calCount);
        #elif VF == 13
            FusedMulAddRelu(dstLocalX, srcLocalX, srcLocalY, calCount);
        #elif VF == 14
            T scalar = srcLocalX.GetValue(0);
            MulsCast(dstLocalX, scalar, srcLocalY, calCount);
        #elif VF == 15
            T scalar = srcLocalY.GetValue(0);
            MulsCast(dstLocalX, srcLocalX, scalar, calCount);
        #elif VF == 16
            AbsSub(dstLocalX, srcLocalX, srcLocalY, calCount);
        #elif VF == 17
            MulAddRelu(dstLocalX, srcLocalX, srcLocalY, calCount);
        #elif VF == 18
            ExpSub(dstLocalX, srcLocalX, srcLocalY, calCount);
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
 
struct MicroVecBinaryParams {
    void (*CallFunc)();
};
 
template<typename T, typename U, int16_t VF>
void RunCase() {
    int src_byte_size = sizeof(T);
    int dst_byte_size = sizeof(U);
    int data_size = 256;
    int mask = 256;
    uint8_t dstGm0[data_size * dst_byte_size] = {0};
    uint8_t dstGm1[data_size * dst_byte_size] = {0};
    uint8_t srcGm0[data_size * src_byte_size] = {0};
    uint8_t srcGm1[data_size * src_byte_size] = {0};
 
    KernelVecBinary<T, U, VF> op;
    op.Init(srcGm0, srcGm1, dstGm0, dstGm1, data_size, mask);
    op.Process();
}

class MicroVecBinaryTestSuite : public testing::Test, public testing::WithParamInterface<MicroVecBinaryParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};
 
INSTANTIATE_TEST_CASE_P(MicroVecBinaryTestCases, MicroVecBinaryTestSuite,
    ::testing::Values(MicroVecBinaryParams { RunCase<float, float, 0>},
                      MicroVecBinaryParams { RunCase<half, half, 1>},
                      MicroVecBinaryParams { RunCase<float, float, 1>},
                      MicroVecBinaryParams { RunCase<int32_t, int32_t, 2>},
                      MicroVecBinaryParams { RunCase<uint32_t, uint32_t, 2>},
                      MicroVecBinaryParams { RunCase<half, half, 3>},
                      MicroVecBinaryParams { RunCase<float, float, 3>},
                      MicroVecBinaryParams { RunCase<half, float, 4>},
                      MicroVecBinaryParams { RunCase<float, float, 4>},
                      MicroVecBinaryParams { RunCase<float, half, 5>},
                      MicroVecBinaryParams { RunCase<float, half, 6>},
                      MicroVecBinaryParams { RunCase<uint64_t, uint64_t, 7>},
                      MicroVecBinaryParams { RunCase<int64_t, int64_t, 7>},
                      MicroVecBinaryParams { RunCase<int64_t, int64_t, 8>},
                      MicroVecBinaryParams { RunCase<uint64_t, uint64_t, 9>},
                      MicroVecBinaryParams { RunCase<int64_t, int64_t, 9>},
                      MicroVecBinaryParams { RunCase<int64_t, int64_t, 10>},
                      MicroVecBinaryParams { RunCase<uint64_t, uint64_t, 11>},
                      MicroVecBinaryParams { RunCase<int64_t, int64_t, 11>},
                      MicroVecBinaryParams { RunCase<int64_t, int64_t, 12>},
                      MicroVecBinaryParams { RunCase<uint64_t, uint64_t, 13>},
                      MicroVecBinaryParams { RunCase<int64_t, int64_t, 13>},
                      MicroVecBinaryParams { RunCase<float, half, 14>},
                      MicroVecBinaryParams { RunCase<float, half, 15>},
                      MicroVecBinaryParams { RunCase<half, half, 16>},
                      MicroVecBinaryParams { RunCase<float, float, 16>},
                      MicroVecBinaryParams { RunCase<uint64_t, uint64_t, 17>},
                      MicroVecBinaryParams { RunCase<int64_t, int64_t, 17>},
                      MicroVecBinaryParams { RunCase<half, float, 18>},
                      MicroVecBinaryParams { RunCase<float, float, 18>}
                      ));
 
TEST_P(MicroVecBinaryTestSuite, MicroVecBinaryTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}