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
class ScatterTest {
public:
    __aicore__ inline ScatterTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
        __gm__ uint8_t* indexOffsetGm, const uint32_t count)
    {
        mElementCount = count;
        mSrcGlobal.SetGlobalBuffer((__gm__ SrcT*)srcGm);
        mIndexOffsetGlobal.SetGlobalBuffer((__gm__ IndexT*)indexOffsetGm);
        mDstGlobal.SetGlobalBuffer((__gm__ DstT*)dstGm);

        mPipe.InitBuffer(mQueIn, 1, mElementCount * sizeof(SrcT));
        mPipe.InitBuffer(mQueIn2, 1, mElementCount * sizeof(IndexT));
        mPipe.InitBuffer(mQueOut, 1, mElementCount * sizeof(DstT));
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
        LocalTensor<SrcT> srcLocal = mQueIn.AllocTensor<SrcT>();
        DataCopy(srcLocal, mSrcGlobal, mElementCount);
        mQueIn.EnQue(srcLocal);

        LocalTensor<IndexT> indexLocal = mQueIn2.AllocTensor<IndexT>();
        DataCopy(indexLocal, mIndexOffsetGlobal, mElementCount);
        mQueIn2.EnQue(indexLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<SrcT> srcLocal = mQueIn.DeQue<SrcT>();
        LocalTensor<IndexT> indexLocal = mQueIn2.DeQue<IndexT>();
        LocalTensor<DstT> dstLocal = mQueOut.AllocTensor<DstT>();
        __ubuf__ SrcT* srcPtr = (__ubuf__ SrcT*)srcLocal.GetPhyAddr();
        __ubuf__ IndexT* indexPtr = (__ubuf__ IndexT*)indexLocal.GetPhyAddr();
        __ubuf__ DstT* dstPtr = (__ubuf__ DstT*)dstLocal.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::RegTensor<IndexT> vIndexReg;
            Reg::RegTensor<SrcT> vSrcReg;
            uint32_t sreg = static_cast<uint32_t>(mElementCount);
            Reg::MaskReg preg;
            uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(SrcT);
            uint16_t repeatTimes = CeilDivision(mElementCount, repeatElm);
            for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
                preg = Reg::UpdateMask<SrcT>(sreg);
                Reg::DataCopy(vIndexReg, indexPtr + i * repeatElm);
                Reg::DataCopyScatter(dstPtr, vSrcReg, vIndexReg, preg);
            }
        }

        mQueIn.FreeTensor(srcLocal);
        mQueIn2.FreeTensor(indexLocal);
        mQueOut.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<DstT> dstLocal = mQueOut.DeQue<DstT>();
        DataCopy(mDstGlobal, dstLocal, mElementCount);
        mQueOut.FreeTensor(dstLocal);
    }
private:
    TPipe mPipe;
    uint32_t mElementCount;
    GlobalTensor<SrcT> mSrcGlobal;
    GlobalTensor<IndexT> mIndexOffsetGlobal;
    GlobalTensor<DstT> mDstGlobal;
    TQue<TPosition::VECIN, 1> mQueIn;
    TQue<TPosition::VECIN, 1> mQueIn2;
    TQue<TPosition::VECOUT, 1> mQueOut;
};
} // namespace AscendC

template <typename DstT, typename SrcT, typename IndexT>
__global__ __aicore__ void testScatter(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, __gm__ uint8_t* indexOffsetGm, uint32_t elementCount)
{
    AscendC::ScatterTest<DstT, SrcT, IndexT> op;
    op.Init(dstGm, srcGm, indexOffsetGm, elementCount);
    op.Process();
}

struct ScatterParams {
    uint32_t dstTypeSize;
    uint32_t srcTypeSize;
    uint32_t indexTypeSize;
    uint32_t elementCount;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint32_t);
};

class ScatterTestsuite : public testing::Test,
    public testing::WithParamInterface<ScatterParams> {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_GATHER, ScatterTestsuite,
    ::testing::Values(
        ScatterParams { 1, 1, 2, 128, testScatter<int8_t, int8_t, uint16_t>},
        ScatterParams { 1, 1, 2, 128, testScatter<uint8_t, uint8_t, uint16_t>},
        ScatterParams { 2, 2, 2, 128, testScatter<int16_t, int16_t, uint16_t>},
        ScatterParams { 2, 2, 2, 128, testScatter<uint16_t, uint16_t, uint16_t>},
        ScatterParams { 4, 4, 4, 128, testScatter<int32_t, int32_t, uint32_t>},
        ScatterParams { 4, 4, 4, 128, testScatter<uint32_t, uint32_t, uint32_t>},
        ScatterParams { 2, 2, 2, 128, testScatter<half, half, uint16_t>},
        ScatterParams { 4, 4, 4, 128, testScatter<float, float, uint32_t>},
        ScatterParams { 2, 2, 2, 128, testScatter<bfloat16_t, bfloat16_t, uint16_t>}));


TEST_P(ScatterTestsuite, testScatter)
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