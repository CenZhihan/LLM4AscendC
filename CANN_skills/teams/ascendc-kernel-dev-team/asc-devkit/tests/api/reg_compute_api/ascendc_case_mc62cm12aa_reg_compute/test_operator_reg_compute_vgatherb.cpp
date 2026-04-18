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
template <typename DstT, typename SrcT, typename IndexT>
class GatherbTest {
public:
    __aicore__ inline GatherbTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
        __gm__ uint8_t* indexOffsetGm, const uint32_t count)
    {
        m_elementCount = count;
        m_srcGlobal.SetGlobalBuffer((__gm__ SrcT*)srcGm);
        m_indexOffsetGlobal.SetGlobalBuffer((__gm__ IndexT*)indexOffsetGm);
        m_dstGlobal.SetGlobalBuffer((__gm__ DstT*)dstGm);

        m_pipe.InitBuffer(m_queIn, 1, m_elementCount * sizeof(SrcT));
        m_pipe.InitBuffer(m_queIn2, 1, m_elementCount * sizeof(IndexT));
        m_pipe.InitBuffer(m_queOut, 1, m_elementCount * sizeof(DstT));
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
        LocalTensor<SrcT> srcLocal = m_queIn.AllocTensor<SrcT>();
        DataCopy(srcLocal, m_srcGlobal, m_elementCount);
        m_queIn.EnQue(srcLocal);

        LocalTensor<IndexT> indexLocal = m_queIn2.AllocTensor<IndexT>();
        DataCopy(indexLocal, m_indexOffsetGlobal, m_elementCount);
        m_queIn2.EnQue(indexLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<SrcT> srcLocal = m_queIn.DeQue<SrcT>();
        LocalTensor<IndexT> indexLocal = m_queIn2.DeQue<IndexT>();
        LocalTensor<DstT> dstLocal = m_queOut.AllocTensor<DstT>();
        __ubuf__ SrcT* srcPtr = (__ubuf__ SrcT*)srcLocal.GetPhyAddr();
        __ubuf__ IndexT* indexPtr = (__ubuf__ IndexT*)indexLocal.GetPhyAddr();
        __ubuf__ DstT* dstPtr = (__ubuf__ DstT*)dstLocal.GetPhyAddr();

        using Reg::RegTensor;
        using Reg::MaskReg;
        using Reg::CreateMask;
        using Reg::UpdateMask;

        __VEC_SCOPE__
        {
            RegTensor<IndexT> vIndexReg;
            RegTensor<DstT> vDstReg;
            uint32_t sreg = (uint32_t)m_elementCount;
            MaskReg preg;
            uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(SrcT);
            uint16_t repeatTimes = CeilDivision(m_elementCount, repeatElm);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                if constexpr (sizeof(SrcT) == 8) {
                    preg = CreateMask<uint8_t, Reg::MaskPattern::ALL>();  // notice in gatherb instruction regarded as B32 format
                } else {
                    preg = UpdateMask<SrcT>(sreg);
                }
                Reg::DataCopy(vIndexReg, indexPtr + i * repeatElm);

                DataCopyGatherB(vDstReg, srcPtr, vIndexReg, preg);
                if constexpr (sizeof(SrcT) == 8) {
                    Reg::DataCopy<SrcT, Reg::StoreDist::DIST_NORM_B32>(dstPtr + i * repeatElm, vDstReg, preg);
                } else {
                    Reg::DataCopy(dstPtr + i * repeatElm, vDstReg, preg);
                }
            }
        }

        m_queIn.FreeTensor(srcLocal);
        m_queIn2.FreeTensor(indexLocal);
        m_queOut.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<DstT> dstLocal = m_queOut.DeQue<DstT>();
        DataCopy(m_dstGlobal, dstLocal, m_elementCount);
        m_queOut.FreeTensor(dstLocal);
    }
private:
    TPipe m_pipe;
    uint32_t m_elementCount;
    GlobalTensor<SrcT> m_srcGlobal;
    GlobalTensor<IndexT> m_indexOffsetGlobal;
    GlobalTensor<DstT> m_dstGlobal;
    TQue<TPosition::VECIN, 1> m_queIn;
    TQue<TPosition::VECIN, 1> m_queIn2;
    TQue<TPosition::VECOUT, 1> m_queOut;
};
} // namespace AscendC

template <typename DstT, typename SrcT, typename IndexT>
__global__ __aicore__ void testGatherb(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, __gm__ uint8_t* indexOffsetGm, uint32_t elementCount)
{
    AscendC::GatherbTest<DstT, SrcT, IndexT> op;
    op.Init(dstGm, srcGm, indexOffsetGm, elementCount);
    op.Process();
}

struct gatherbParams {
    uint32_t dstTypeSize;
    uint32_t srcTypeSize;
    uint32_t indexTypeSize;
    uint32_t elementCount;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint32_t);
};

class GatherbTestsuite : public testing::Test,
    public testing::WithParamInterface<gatherbParams> {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_GATHER, GatherbTestsuite,
    ::testing::Values(
        gatherbParams { 1, 1, 4, 256, testGatherb<int8_t, int8_t, uint32_t>},
        gatherbParams { 1, 1, 4, 256, testGatherb<uint8_t, uint8_t, uint32_t>},
        gatherbParams { 2, 2, 4, 256, testGatherb<int16_t, int16_t, uint32_t>},
        gatherbParams { 2, 2, 4, 256, testGatherb<uint16_t, uint16_t, uint32_t>},
        gatherbParams { 4, 4, 4, 128, testGatherb<int32_t, int32_t, uint32_t>},
        gatherbParams { 4, 4, 4, 128, testGatherb<uint32_t, uint32_t, uint32_t>},
        gatherbParams { 2, 2, 4, 256, testGatherb<half, half, uint32_t>},
        gatherbParams { 4, 4, 4, 128, testGatherb<float, float, uint32_t>},
        gatherbParams { 2, 2, 4, 256, testGatherb<bfloat16_t, bfloat16_t, uint32_t>},
        gatherbParams { 8, 8, 4, 256, testGatherb<uint64_t, uint64_t, uint32_t>},
        gatherbParams { 8, 8, 4, 256, testGatherb<int64_t, int64_t, uint32_t>}));


TEST_P(GatherbTestsuite, testGatherb)
{
    auto param = GetParam();
    uint8_t dstGm[param.elementCount * param.dstTypeSize] = {0};
    uint8_t srcGm[param.elementCount * param.srcTypeSize] = {0};
    uint8_t indexOffsetGm[param.elementCount * param.indexTypeSize] = {0};
    param.cal_func(dstGm, srcGm, indexOffsetGm, param.elementCount);
    
    for (int32_t i = 0; i < param.elementCount; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}