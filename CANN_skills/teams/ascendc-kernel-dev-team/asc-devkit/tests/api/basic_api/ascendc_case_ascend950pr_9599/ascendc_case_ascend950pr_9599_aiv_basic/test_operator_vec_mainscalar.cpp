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

template <typename T, typename U, int16_t VF>
class KernelMainScalar {
public:
    __aicore__ inline KernelMainScalar()
    {}
    __aicore__ inline void Init(GM_ADDR srcGm0, GM_ADDR srcGm1, GM_ADDR dstGm0, GM_ADDR dstGm1, uint32_t dataSizeIn, uint32_t calCountIn)
    {
        calCount = calCountIn;
        const int srcAlignSize = 32 / sizeof(PrimT<T>);
        srcDataSize = CeilDivision(dataSizeIn, srcAlignSize) * srcAlignSize;
        const int dstAlignSize = 32 / sizeof(PrimT<U>);
        dstDataSize = CeilDivision(dataSizeIn, dstAlignSize) * dstAlignSize;
        srcGlobalX.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T> *>(srcGm0), srcDataSize);
        srcGlobalY.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T> *>(srcGm1), srcDataSize);
        dstGlobalX.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<U> *>(dstGm0), dstDataSize);
        dstGlobalY.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<U> *>(dstGm1), dstDataSize);
        pipe.InitBuffer(inQueueX, 1, srcDataSize * sizeof(PrimT<T>));
        pipe.InitBuffer(inQueueY, 1, srcDataSize * sizeof(PrimT<T>));
        pipe.InitBuffer(outQueueX, 1, dstDataSize * sizeof(PrimT<U>));
        pipe.InitBuffer(outQueueY, 1, dstDataSize * sizeof(PrimT<U>));
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
        uint32_t repeatElm = GetVecLen() / sizeof(PrimT<T>);
        uint32_t repeatTimes = srcDataSize / repeatElm;
        LocalTensor<U> dstLocalX = outQueueX.AllocTensor<U>();
        LocalTensor<U> dstLocalY = outQueueY.AllocTensor<U>();
        LocalTensor<T> srcLocalX = inQueueX.DeQue<T>();
        LocalTensor<T> srcLocalY = inQueueY.DeQue<T>();
        PrimT<U> Zero = 0;
        if constexpr (IsSameType<PrimT<T>, int64_t>::value || IsSameType<PrimT<T>, float>::value) {
            Zero = AscendC::Std::sqrt(Zero);
        }
        if constexpr (IsSameType<PrimT<T>, int64_t>::value || IsSameType<PrimT<T>, int32_t>::value || 
                      IsSameType<PrimT<T>, int16_t>::value || IsSameType<PrimT<T>, int8_t>::value || 
                      IsSameType<PrimT<T>, float>::value) {
            Zero = AscendC::Std::abs(Zero);
        }
        if constexpr (IsSameType<PrimT<T>, T>::value) {
            Duplicate(dstLocalX, Zero, dstDataSize);
            Duplicate(dstLocalY, Zero, dstDataSize);
        }
     
        #ifdef VF
        #if VF == 0
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
 
struct MicroMainScalarParams {
    void (*CallFunc)();
};
 
template<typename T, typename U, int16_t VF>
void RunCase() {
    int srcByteSize = sizeof(PrimT<T>);
    int dstByteSize = sizeof(PrimT<U>);
    int dataSize = 256;
    int mask = 256;
    uint8_t dstGm0[dataSize * dstByteSize] = {0};
    uint8_t dstGm1[dataSize * dstByteSize] = {0};
    uint8_t srcGm0[dataSize * srcByteSize] = {0};
    uint8_t srcGm1[dataSize * srcByteSize] = {0};
 
    KernelMainScalar<T, U, VF> op;
    op.Init(srcGm0, srcGm1, dstGm0, dstGm1, dataSize, mask);
    op.Process();
}

class MicroMainScalarTestSuite : public testing::Test, public testing::WithParamInterface<MicroMainScalarParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};
 
INSTANTIATE_TEST_CASE_P(MicroMainScalarTestCases, MicroMainScalarTestSuite,
    ::testing::Values(MicroMainScalarParams { RunCase<float, float, 0>},
                      MicroMainScalarParams { RunCase<int8_t, int8_t, 0>},
                      MicroMainScalarParams { RunCase<int16_t, int16_t, 0>},
                      MicroMainScalarParams { RunCase<int32_t, int32_t, 0>},
                      MicroMainScalarParams { RunCase<int64_t, int64_t, 0>},
                      // TensorTrait
                      MicroMainScalarParams { RunCase<TensorTrait<float>, TensorTrait<float>, 0>},
                      MicroMainScalarParams { RunCase<TensorTrait<int8_t>, TensorTrait<int8_t>, 0>},
                      MicroMainScalarParams { RunCase<TensorTrait<int16_t>, TensorTrait<int16_t>, 0>},
                      MicroMainScalarParams { RunCase<TensorTrait<int32_t>, TensorTrait<int32_t>, 0>},
                      MicroMainScalarParams { RunCase<TensorTrait<int64_t>, TensorTrait<int64_t>, 0>}
                      ));
 
TEST_P(MicroMainScalarTestSuite, MicroMainScalarTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}