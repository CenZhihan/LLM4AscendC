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
template <typename T, int mode>
class MicroCreateIndexTest {
public:
    __aicore__ inline MicroCreateIndexTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
        __gm__ uint8_t* indexOffsetGm, const uint32_t count)
    {
        m_elementCount = count;
        m_srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        m_indexOffsetGlobal.SetGlobalBuffer((__gm__ T*)indexOffsetGm);
        m_dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
 
        m_pipe.InitBuffer(m_queIn, 1, m_elementCount * sizeof(T));
        m_pipe.InitBuffer(m_queIn2, 1, m_elementCount * sizeof(T));
        m_pipe.InitBuffer(m_queOut, 1, m_elementCount * sizeof(T));
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
        LocalTensor<T> srcLocal = m_queIn.AllocTensor<T>();
        DataCopy(srcLocal, m_srcGlobal, m_elementCount);
        m_queIn.EnQue(srcLocal);
 
        LocalTensor<T> indexLocal = m_queIn2.AllocTensor<T>();
        DataCopy(indexLocal, m_indexOffsetGlobal, m_elementCount);
        m_queIn2.EnQue(indexLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = m_queIn.DeQue<T>();
        LocalTensor<T> indexLocal = m_queIn2.DeQue<T>();
        LocalTensor<T> dstLocal = m_queOut.AllocTensor<T>();
        __ubuf__ T* srcPtr = (__ubuf__ T*)srcLocal.GetPhyAddr();
        __ubuf__ T* indexPtr = (__ubuf__ T*)indexLocal.GetPhyAddr();
        __ubuf__ T* dstPtr = (__ubuf__ T*)dstLocal.GetPhyAddr();
 
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> vIndexReg;
            Reg::RegTensor<T> vDstReg;
            Reg::RegTensor<T> vSrcReg;
            uint32_t sreg = (uint32_t)m_elementCount;
            Reg::MaskReg preg;
            uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(T);
            uint16_t repeatTimes = CeilDivision(m_elementCount, repeatElm);
            T scalar = 1;
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                preg = Reg::UpdateMask<half>(sreg);
                Reg::DataCopy(vIndexReg, indexPtr + i * repeatElm);
                Reg::Arange(vDstReg, scalar);
 
                Reg::DataCopy(dstPtr + i * repeatElm, vDstReg, preg);
            }
        }
       
 
        m_queIn.FreeTensor(srcLocal);
        m_queIn2.FreeTensor(indexLocal);
        m_queOut.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = m_queOut.DeQue<T>();
        DataCopy(m_dstGlobal, dstLocal, m_elementCount);
        m_queOut.FreeTensor(dstLocal);
    }
private:
    TPipe m_pipe;
    uint32_t m_elementCount;
    GlobalTensor<T> m_srcGlobal;
    GlobalTensor<T> m_indexOffsetGlobal;
    GlobalTensor<T> m_dstGlobal;
    TQue<TPosition::VECIN, 1> m_queIn;
    TQue<TPosition::VECIN, 1> m_queIn2;
    TQue<TPosition::VECOUT, 1> m_queOut;
};
} // namespace AscendC

template <typename T, int mode>
__global__ __aicore__ void testCreIndex(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
    __gm__ uint8_t* indexOffsetGm, uint32_t elementCount)
{
    AscendC::MicroCreateIndexTest<T, mode> op;
    op.Init(dstGm, srcGm, indexOffsetGm, elementCount);
    op.Process();
}

struct creIndexParams {
    uint32_t dstTypeSize;
    uint32_t srcTypeSize;
    uint32_t elementCount;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint32_t);
};

class creIndexTestsuite : public testing::Test,
    public testing::WithParamInterface<creIndexParams> {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_CREINDEX, creIndexTestsuite,
    ::testing::Values(creIndexParams { 2, 2, 128, testCreIndex<half, 0>},
    creIndexParams { 4, 4, 64, testCreIndex<int32_t, 0>},
    creIndexParams { 2, 2, 128, testCreIndex<int16_t, 0>},
    creIndexParams { 2, 2, 128, testCreIndex<int8_t, 0>},
    creIndexParams { 4, 4, 64, testCreIndex<float, 0>}));


TEST_P(creIndexTestsuite, testCreIndex)
{
    auto param = GetParam();
    uint8_t dstGm[param.elementCount * param.dstTypeSize] = {0};
    uint8_t srcGm[param.elementCount * param.srcTypeSize] = {0};
    uint8_t indexOffsetGm[param.elementCount * param.dstTypeSize] = {0};
    param.cal_func(dstGm, srcGm, indexOffsetGm, param.elementCount);
    
    for (int32_t i = 0; i < param.elementCount; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}