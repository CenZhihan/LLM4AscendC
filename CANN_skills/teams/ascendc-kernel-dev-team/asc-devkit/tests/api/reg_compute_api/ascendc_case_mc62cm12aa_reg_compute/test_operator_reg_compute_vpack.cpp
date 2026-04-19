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


#define DType half
#define DType1 half
#define Mode 8

template <typename DstT, typename SrcT, int32_t MD>
class KernelVpack {
public:
    __aicore__ inline KernelVpack()
    {}
    __aicore__ inline void Init(GM_ADDR dst0_gm, GM_ADDR dst1_gm, GM_ADDR src0_gm, GM_ADDR src1_gm,
            uint32_t nums, uint32_t vec_mask)
    {
        src1_global.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT *>(src1_gm), nums);
        dst1_global.SetGlobalBuffer(reinterpret_cast<__gm__ DstT *>(dst1_gm), nums);

        pipe.InitBuffer(inQueueX2, 1, nums * sizeof(SrcT));
        pipe.InitBuffer(outQueue2, 1, nums * sizeof(DstT));
        dataSize = nums;
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
        LocalTensor<SrcT> src1Local = inQueueX2.AllocTensor<SrcT>();
        DataCopy(src1Local, src1_global, dataSize);
        inQueueX2.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<DstT> dst1Local = outQueue2.AllocTensor<DstT>();
        DstT zero = 0;
        LocalTensor<SrcT> src1Local = inQueueX2.DeQue<SrcT>();
        uint16_t mask_bit_size = 256;
        uint16_t one_rep_size = mask_bit_size/sizeof(DstT);
        uint16_t rep = dataSize/one_rep_size;
        __ubuf__ DstT* dst1Ptr = (__ubuf__ DstT*)dst1Local.GetPhyAddr();
        __ubuf__ SrcT* src1Ptr = (__ubuf__ SrcT*)src1Local.GetPhyAddr();
        __VEC_SCOPE__
        {
            Reg::RegTensor<SrcT> vSrcReg1;
            Reg::RegTensor<DstT> vDstReg1;
            uint32_t sreg = (uint32_t)mask;
            Reg::MaskReg maskReg;
            maskReg = Reg::UpdateMask<DstT>(sreg);
            for (uint16_t i = 0; i < (uint16_t)rep; i++) {
                DataCopy(vSrcReg1, src1Ptr + i * one_rep_size);
                if constexpr (MD == 0) {
                    Pack(vDstReg1, vSrcReg1);
                } else if constexpr (MD == 1) {
                    Reg::UnPack(vDstReg1, vSrcReg1);
                }
                DataCopy(dst1Ptr + i * one_rep_size, vDstReg1, maskReg);
            }
        }
        outQueue2.EnQue<DstT>(dst1Local);
        inQueueX2.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<DstT> dst1Local = outQueue2.DeQue<DstT>();
        DataCopy(dst1_global, dst1Local, dataSize);
        outQueue2.FreeTensor(dst1Local);
    }

private:
    GlobalTensor<SrcT> src1_global;
    GlobalTensor<DstT> dst1_global;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX2;
    TQue<TPosition::VECOUT, 1> outQueue2;
    uint32_t dataSize = 0;
    uint32_t mask;
};

struct MicroVpackParams {
    void (*CallFunc)();
};

template<typename DstT, typename SrcT, int32_t mode>
void RunCase() {
    int byte_size = sizeof(DstT);
    int shape_size = 1024;
    int mask = 256;
    int dataSize = 1024;
    uint8_t dst0Gm[shape_size * byte_size] = {0};
    uint8_t dst1Gm[shape_size * byte_size] = {0};
    uint8_t src0Gm[shape_size * byte_size] = {0};
    uint8_t src1Gm[shape_size * byte_size] = {0};

    KernelVpack<DstT, SrcT, mode> op;
    op.Init(dst0Gm, dst1Gm, src0Gm, src1Gm, dataSize, mask);
    op.Process();
}

class MicroVpackTestsuite : public testing::Test, public testing::WithParamInterface<MicroVpackParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(MicroVpackTestCase, MicroVpackTestsuite,
    ::testing::Values(
                    MicroVpackParams { RunCase<uint8_t, uint16_t, 0> },
                    MicroVpackParams { RunCase<uint8_t, int16_t, 0> },
                    MicroVpackParams { RunCase<uint16_t, uint32_t, 0> },
                    MicroVpackParams { RunCase<uint16_t, int32_t, 0> },
                    MicroVpackParams { RunCase<uint32_t, uint16_t, 1> },
                    MicroVpackParams { RunCase<int32_t, int16_t, 1> },
                    MicroVpackParams { RunCase<uint16_t, uint8_t, 1> },
                    MicroVpackParams { RunCase<int16_t, int8_t, 1> }
                        ));

TEST_P(MicroVpackTestsuite, MicroVpackTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}
