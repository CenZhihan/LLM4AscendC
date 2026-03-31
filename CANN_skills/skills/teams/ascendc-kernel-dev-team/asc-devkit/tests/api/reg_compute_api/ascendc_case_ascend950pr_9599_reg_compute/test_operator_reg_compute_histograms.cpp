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
 
template<typename T, typename U, Reg::HistogramsBinType mode, Reg::HistogramsType type>
class KernelHistograms {
public:
    __aicore__ inline KernelHistograms()
    {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t srcSizeIn, uint32_t dstSizeIn, uint32_t vecMask)
    { 
        srcSize = srcSizeIn;
        dstSize = dstSizeIn;
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), srcSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(dstGm), dstSize);
 
        pipe.InitBuffer(inQueue, 1, srcSize * sizeof(T));
        pipe.InitBuffer(outQueue, 1, dstSize * sizeof(U));
       
        mask = vecMask;
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
        DataCopy(srcLocal, srcGlobal, srcSize);
        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<U> dstLocal = outQueue.AllocTensor<U>();
        U zero = 0;
        Duplicate(dstLocal, zero, dstSize);
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        uint16_t maskBitSize = 256;
        uint16_t oneRepSize = maskBitSize/sizeof(T);
        uint16_t rep = srcSize/oneRepSize;
        __ubuf__ U* dstPtr = (__ubuf__ U*)dstLocal.GetPhyAddr();
        __ubuf__ T* srcPtr = (__ubuf__ T*)srcLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> vSrcReg;
            Reg::RegTensor<U> vDstReg;
            uint32_t sreg = static_cast<uint32_t>(mask);
            Reg::MaskReg maskReg;
            maskReg = Reg::UpdateMask<T>(sreg);
            Reg::Histograms<T, U, mode, type>(vDstReg, vSrcReg, maskReg);
        }
        outQueue.EnQue<U>(dstLocal);
        inQueue.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<U> dstLocal = outQueue.DeQue<U>();
        DataCopy(dstGlobal, dstLocal, dstSize);
        outQueue.FreeTensor(dstLocal);
    }
 
private:
    GlobalTensor<T> srcGlobal;
    GlobalTensor<U> dstGlobal;
 
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t srcSize = 0;
    uint32_t dstSize = 0;
    uint32_t mask;
};
 
struct MicroHistogramsParams {
    void (*CallFunc)();
};
 
template<typename T, typename U, Reg::HistogramsBinType mode, Reg::HistogramsType type>
void RunCase() {
    int srcByteSize = sizeof(T);
    int dstByteSize = sizeof(U);
    int dstShapeSize = 256;
    int srcShapeSize = 128;
    int mask = 256;
    uint8_t dstGm[dstShapeSize * dstByteSize] = {0};
    uint8_t srcGm[srcShapeSize * srcByteSize] = {0};
 
    KernelHistograms<T, U, mode, type> op;
    op.Init(dstGm, srcGm, srcShapeSize, dstShapeSize, mask);
    op.Process();
}
 
class MicroHistogramsTestsuite : public testing::Test, public testing::WithParamInterface<MicroHistogramsParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};
 
INSTANTIATE_TEST_CASE_P(MicroCmpTestCase, MicroHistogramsTestsuite,
    ::testing::Values(MicroHistogramsParams { RunCase<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::FREQUENCY>},
                      MicroHistogramsParams { RunCase<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::FREQUENCY>},
                      MicroHistogramsParams { RunCase<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>},
                      MicroHistogramsParams { RunCase<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>}
                      ));
 
TEST_P(MicroHistogramsTestsuite, MicroHistogramsTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}