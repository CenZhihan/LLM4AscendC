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
    __aicore__ inline void Init(GM_ADDR dst_gm, GM_ADDR src_gm, uint32_t srcSizeIn, uint32_t dstSizeIn, uint32_t vec_mask)
    { 
        srcSize = srcSizeIn;
        dstSize = dstSizeIn;
        src_global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src_gm), srcSize);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(dst_gm), dstSize);
 
        pipe.InitBuffer(inQueue, 1, srcSize * sizeof(T));
        pipe.InitBuffer(outQueue, 1, dstSize * sizeof(U));
       
        mask = vec_mask;
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
        DataCopy(srcLocal, src_global, srcSize);
        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<U> dstLocal = outQueue.AllocTensor<U>();
        U zero = 0;
        Duplicate(dstLocal, zero, dstSize);
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        uint16_t mask_bit_size = 256;
        uint16_t one_rep_size = mask_bit_size/sizeof(T);
        uint16_t rep = srcSize/one_rep_size;
        __ubuf__ U* dstPtr = (__ubuf__ U*)dstLocal.GetPhyAddr();
        __ubuf__ T* srcPtr = (__ubuf__ T*)srcLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> vSrcReg;
            Reg::RegTensor<U> vDstReg;
            uint32_t sreg = (uint32_t)mask;
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
        DataCopy(dst_global, dstLocal, dstSize);
        outQueue.FreeTensor(dstLocal);
    }
 
private:
    GlobalTensor<T> src_global;
    GlobalTensor<U> dst_global;
 
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
    int src_byte_size = sizeof(T);
    int dst_byte_size = sizeof(U);
    int dst_shape_size = 256;
    int src_shape_size = 128;
    int mask = 256;
    uint8_t dstGm[dst_shape_size * dst_byte_size] = {0};
    uint8_t srcGm[src_shape_size * src_byte_size] = {0};
 
    KernelHistograms<T, U, mode, type> op;
    op.Init(dstGm, srcGm, src_shape_size, dst_shape_size, mask);
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