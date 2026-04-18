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

template <typename SrcType, typename DstType>
class KernelVcvt {

public:
    __aicore__ inline KernelVcvt()
    {}
    __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t srcSize, uint32_t dstSize)
    {
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ SrcType *>(srcGm), srcSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstType *>(dstGm), dstSize);

        pipe.InitBuffer(inQueueX, 1, srcSize * sizeof(SrcType));
        pipe.InitBuffer(outQueue, 1, dstSize * sizeof(DstType));
        srcDataSize = srcSize;
        dstDataSize = dstSize;
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
        LocalTensor<SrcType> srcLocal = inQueueX.AllocTensor<SrcType>();
        DataCopy(srcLocal, srcGlobal, srcDataSize);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<DstType> dstLocal = outQueue.AllocTensor<DstType>();

        LocalTensor<SrcType> srcLocal = inQueueX.DeQue<SrcType>();
        uint32_t oneRepSize = 64;
        uint32_t rep = srcDataSize/oneRepSize;
        __VEC_SCOPE__
        {
            Reg::RegTensor<SrcType> vreg0;
            Reg::RegTensor<DstType> vreg1;
            Reg::RegTensor<SrcType> dst0;
            Reg::RegTensor<SrcType> dst1;
            uint32_t sreg =  static_cast<uint32_t>(dstDataSize);
            constexpr uint32_t sregLower =  static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(float));
            Reg::MaskReg preg1;
            __ubuf__ DstType* dstPtr = (__ubuf__ DstType*)dstLocal.GetPhyAddr();
            __ubuf__ SrcType* srcPtr = (__ubuf__ SrcType*)srcLocal.GetPhyAddr();
            uint32_t dist = 2;
            Reg::MaskReg preg = Reg::CreateMask<DstType>();
            for (uint16_t i = 0; i < rep; i++) {
                preg1 = Reg::UpdateMask<float>(sreg);
                Reg::DataCopy(vreg0, srcPtr + i * sregLower);
                vintlv(dst0, dst1, vreg0, vreg0);
                vcvt(vreg1, dst0, preg1, PART_EVEN, MODE_ZEROING);
                vcvt(vreg0, vreg1, preg1, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                vdintlv(dst0, dst1, vreg0, vreg0);
            }
        }
        outQueue.EnQue<DstType>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<DstType> dstLocal = outQueue.DeQue<DstType>();
        DataCopy(dstGlobal, dstLocal, dstDataSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<SrcType> srcGlobal;
    GlobalTensor<DstType> dstGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t dstDataSize = 0;
    uint32_t srcDataSize = 0;
};

struct VcvtParams {
    int32_t srcSize;
    int32_t dstSize;
};

class VcvtTestsuite : public testing::Test, public testing::WithParamInterface<VcvtParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(VcvtTestCase, VcvtTestsuite,
    ::testing::Values(VcvtParams { 128, 64 }));

TEST_P(VcvtTestsuite, VcvtTestCase)
{
    auto param = GetParam();
    int srcSize = param.srcSize;
    int dstSize = param.dstSize;
    int halfByteSize = 2;
    int fpByteSize = 4;
    uint8_t srcGm[srcSize * halfByteSize] = {0};
    uint8_t dstGm[dstSize * fpByteSize] = {0};
    KernelVcvt<half, float> op;
    op.Init(srcGm, dstGm, srcSize, dstSize);
    op.Process();
}