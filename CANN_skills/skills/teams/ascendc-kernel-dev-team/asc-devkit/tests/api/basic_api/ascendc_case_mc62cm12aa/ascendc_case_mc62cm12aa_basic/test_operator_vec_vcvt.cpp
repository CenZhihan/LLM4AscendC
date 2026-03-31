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

template <typename srcType, typename dstType>
class KernelVcvt {

public:
    __aicore__ inline KernelVcvt()
    {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, uint32_t srcSize, uint32_t dstSize)
    {
        src_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src_gm), srcSize);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ dstType *>(dst_gm), dstSize);

        pipe.InitBuffer(inQueueX, 1, srcSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, dstSize * sizeof(dstType));
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
        LocalTensor<srcType> srcLocal = inQueueX.AllocTensor<srcType>();
        DataCopy(srcLocal, src_global, srcDataSize);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<dstType> dstLocal = outQueue.AllocTensor<dstType>();

        LocalTensor<srcType> srcLocal = inQueueX.DeQue<srcType>();
        uint32_t one_rep_size = 64;
        uint32_t rep = srcDataSize/one_rep_size;
        __VEC_SCOPE__
        {
            Reg::RegTensor<srcType> vreg0;
            Reg::RegTensor<dstType> vreg1;
            Reg::RegTensor<srcType> dst0;
            Reg::RegTensor<srcType> dst1;
            uint32_t sreg = (uint32_t)dstDataSize;
            constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(float));
            Reg::MaskReg preg1;
            __ubuf__ dstType* dstPtr = (__ubuf__ dstType*)dstLocal.GetPhyAddr();
            __ubuf__ srcType* srcPtr = (__ubuf__ srcType*)srcLocal.GetPhyAddr();
            uint32_t dist = 2;
            Reg::MaskReg preg = Reg::CreateMask<dstType>();
            for (uint16_t i = 0; i < rep; i++) {
                preg1 = Reg::UpdateMask<float>(sreg);
                Reg::DataCopy(vreg0, srcPtr + i * sregLower);
                vintlv(dst0, dst1, vreg0, vreg0);
                vcvt(vreg1, dst0, preg1, PART_EVEN, MODE_ZEROING);
                vcvt(vreg0, vreg1, preg1, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                vdintlv(dst0, dst1, vreg0, vreg0);
            }
        }
        outQueue.EnQue<dstType>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<dstType> dstLocal = outQueue.DeQue<dstType>();
        DataCopy(dst_global, dstLocal, dstDataSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<srcType> src_global;
    GlobalTensor<dstType> dst_global;

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
    int half_byte_size = 2;
    int fp_byte_size = 4;
    uint8_t srcGm[srcSize * half_byte_size] = {0};
    uint8_t dstGm[dstSize * fp_byte_size] = {0};
    KernelVcvt<half, float> op;
    op.Init(srcGm, dstGm, srcSize, dstSize);
    op.Process();
}