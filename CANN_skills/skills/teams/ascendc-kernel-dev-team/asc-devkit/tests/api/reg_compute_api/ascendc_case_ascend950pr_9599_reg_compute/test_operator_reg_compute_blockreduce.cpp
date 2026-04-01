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

namespace AscendC {
template <typename T, typename U, int mode>
class MicroBlockReduceTest {
public:
    __aicore__ inline MicroBlockReduceTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
        const uint32_t count)
    {
        elementCount = count;
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
 
        pipe.InitBuffer(queIn, 1, elementCount * sizeof(T));
        pipe.InitBuffer(queOut, 1, elementCount * sizeof(T));
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
        LocalTensor<T> srcLocal = queIn.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, elementCount);
        queIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = queIn.DeQue<T>();
        LocalTensor<T> dstLocal = queOut.AllocTensor<T>();
        __ubuf__ T* srcPtr = (__ubuf__ T*)srcLocal.GetPhyAddr();
        __ubuf__ T* dstPtr = (__ubuf__ T*)dstLocal.GetPhyAddr();
 
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> vDstReg;
            Reg::RegTensor<U> vSrcReg;
            uint32_t sreg = static_cast<uint32_t>(elementCount);
            Reg::MaskReg preg;
            uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(T);
            uint16_t repeatTimes = CeilDivision(elementCount, repeatElm);
            T scalar = 1;
            for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
                preg = Reg::UpdateMask<T>(sreg);
                Reg::DataCopy(vSrcReg, srcPtr + i * repeatElm);
                if constexpr(mode == 0) {
                    Reg::ReduceSumWithDataBlock(vDstReg, vSrcReg, preg);
                } else if constexpr(mode == 1) {
                    Reg::ReduceMaxWithDataBlock(vDstReg, vSrcReg, preg);
                } else if constexpr(mode == 2) {
                    Reg::ReduceMinWithDataBlock(vDstReg, vSrcReg, preg);
                }
                Reg::DataCopy(dstPtr + i * repeatElm, vDstReg, preg);
            }
        }
       
 
        queIn.FreeTensor(srcLocal);
        queOut.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = queOut.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, elementCount);
        queOut.FreeTensor(dstLocal);
    }
private:
    TPipe pipe;
    uint32_t elementCount;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    TQue<TPosition::VECIN, 1> queIn;
    TQue<TPosition::VECOUT, 1> queOut;
};
} // namespace AscendC

template <typename T, typename U, int mode>
__global__ __aicore__ void testBlockReduce(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
    uint32_t elementCount)
{
    AscendC::MicroBlockReduceTest<T, U, mode> op;
    op.Init(dstGm, srcGm, elementCount);
    op.Process();
}

struct BlockReduceParams {
    uint32_t dstTypeSize;
    uint32_t srcTypeSize;
    uint32_t elementCount;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t);
};

class blockReduceTestsuite : public testing::Test,
    public testing::WithParamInterface<BlockReduceParams> {};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_BLOCK_REDUCE, blockReduceTestsuite,
    ::testing::Values(BlockReduceParams { 4, 4, 128, testBlockReduce<float, float, 0>},
    BlockReduceParams { 4, 4, 128, testBlockReduce<uint32_t, uint32_t, 0>},
    BlockReduceParams { 4, 4, 128, testBlockReduce<int32_t, int32_t, 0>},
    BlockReduceParams { 2, 2, 256, testBlockReduce<int16_t, int16_t, 0>},
    BlockReduceParams { 2, 2, 256, testBlockReduce<uint16_t, uint16_t, 0>},
    BlockReduceParams { 2, 2, 256, testBlockReduce<half, half, 0>},
    BlockReduceParams { 4, 4, 128, testBlockReduce<float, float, 1>},
    BlockReduceParams { 4, 4, 128, testBlockReduce<uint32_t, uint32_t, 1>},
    BlockReduceParams { 4, 4, 128, testBlockReduce<int32_t, int32_t, 1>},
    BlockReduceParams { 2, 2, 256, testBlockReduce<int16_t, int16_t, 1>},
    BlockReduceParams { 2, 2, 256, testBlockReduce<uint16_t, uint16_t, 1>},
    BlockReduceParams { 2, 2, 256, testBlockReduce<half, half, 1>},
    BlockReduceParams { 4, 4, 128, testBlockReduce<float, float, 2>},
    BlockReduceParams { 4, 4, 128, testBlockReduce<uint32_t, uint32_t, 2>},
    BlockReduceParams { 4, 4, 128, testBlockReduce<int32_t, int32_t, 2>},
    BlockReduceParams { 2, 2, 256, testBlockReduce<int16_t, int16_t, 2>},
    BlockReduceParams { 2, 2, 256, testBlockReduce<uint16_t, uint16_t, 2>},
    BlockReduceParams { 2, 2, 256, testBlockReduce<half, half, 2>}));

TEST_P(blockReduceTestsuite, testBlockReduce)
{
    auto param = GetParam();
    uint8_t dstGm[param.elementCount * param.dstTypeSize] = {0};
    uint8_t srcGm[param.elementCount * param.srcTypeSize] = {0};
    param.cal_func(dstGm, srcGm, param.elementCount);
    
    for (int32_t i = 0; i < param.elementCount; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}