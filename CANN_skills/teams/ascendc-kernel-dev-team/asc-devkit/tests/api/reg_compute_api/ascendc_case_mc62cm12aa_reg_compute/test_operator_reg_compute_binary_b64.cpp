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

template <typename T, typename SCR1_T, int32_t MD, int32_t VF>
class KernelBinary {
public:
    __aicore__ inline KernelBinary()
    {}
    __aicore__ inline void Init(GM_ADDR dst0_gm, GM_ADDR dst1_gm, GM_ADDR src0_gm, GM_ADDR src1_gm,
            uint32_t nums, uint32_t vec_mask)
    {
        src0_global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src0_gm), nums);
        src1_global.SetGlobalBuffer(reinterpret_cast<__gm__ SCR1_T *>(src1_gm), nums);
        dst0_global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst0_gm), nums);
        dst1_global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst1_gm), nums);

        pipe.InitBuffer(inQueueX, 1, nums * sizeof(T));
        pipe.InitBuffer(inQueueX2, 1, nums * sizeof(SCR1_T));
        pipe.InitBuffer(outQueue, 1, nums * sizeof(T));
        pipe.InitBuffer(outQueue2, 1, nums * sizeof(T));
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
        LocalTensor<T> src0Local = inQueueX.AllocTensor<T>();
        LocalTensor<SCR1_T> src1Local = inQueueX2.AllocTensor<SCR1_T>();
        DataCopy(src0Local, src0_global, dataSize);
        DataCopy(src1Local, src1_global, dataSize);
        inQueueX.EnQue(src0Local);
        inQueueX2.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> src0Local = inQueueX.DeQue<T>();
        LocalTensor<SCR1_T> src1Local = inQueueX2.DeQue<SCR1_T>();

        // Mirco Trait2
        if constexpr(MD == 0) {
            __ubuf__ T *src0 = (__ubuf__ T *)src0Local.GetPhyAddr();
            __ubuf__ T *src1 = (__ubuf__ T *)src1Local.GetPhyAddr();
            __ubuf__ T *dst = (__ubuf__ T *)dstLocal.GetPhyAddr();
            uint16_t mask_bit_size = 256;
            uint16_t one_rep_size = mask_bit_size/sizeof(T);
            uint16_t repeat_times;
            repeat_times = CeilDivision(dataSize, 64);

            __VEC_SCOPE__ 
            {
                Reg::RegTensor<T, Reg::RegTraitNumTwo> vSrcReg0;
                Reg::RegTensor<T, Reg::RegTraitNumTwo> vSrcReg1;
                Reg::RegTensor<T, Reg::RegTraitNumTwo> vDstReg0;

                uint32_t sreg = (uint32_t)dataSize;
                for (uint16_t i = 0; i < repeat_times; i++) {
                    Reg::MaskReg maskReg = Reg::UpdateMask<T, Reg::RegTraitNumTwo>(sreg);
                    DataCopy(vSrcReg0, src0 + i * one_rep_size * 2);
                    DataCopy(vSrcReg1, src1 + i * one_rep_size * 2);
                    #ifdef VF
                    #if VF == 0
                        Reg::Div(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                    #elif VF == 1:
                        Reg::MulAddDst(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                    #endif
                    #endif
                    DataCopy(dst + i * one_rep_size * 2, vDstReg0, maskReg);
                }
            }
        // Basic
        } else if constexpr(MD == 1) {
            #ifdef VF
            #if VF == 0
                Div(dstLocal, src0Local, src1Local, dataSize);
            #elif VF == 1:
                Reg::MulAddDst(dstLocal, src0Local, src1Local, dataSize);
            #elif VF == 2:
                    Reg::FusedAbsSub(dstLocal, src0Local, src1Local, dataSize);s
            #endif
            #endif
        // Mirco Trait1
        } else if constexpr(MD == 2) {
            __ubuf__ T *src0 = (__ubuf__ T *)src0Local.GetPhyAddr();
            __ubuf__ T *src1 = (__ubuf__ T *)src1Local.GetPhyAddr();
            __ubuf__ T *dst = (__ubuf__ T *)dstLocal.GetPhyAddr();
            uint16_t mask_bit_size = 256;
            uint16_t one_rep_size = mask_bit_size/sizeof(T);
            uint16_t repeat_times;
            repeat_times = CeilDivision(dataSize, one_rep_size);

            __VEC_SCOPE__ 
            {
                Reg::RegTensor<T, Reg::RegTraitNumOne> vSrcReg0;
                Reg::RegTensor<T, Reg::RegTraitNumOne> vSrcReg1;
                Reg::RegTensor<T, Reg::RegTraitNumOne> vDstReg0;

                uint32_t sreg = (uint32_t)dataSize * 2;
                for (uint16_t i = 0; i < repeat_times; i++) {
                    Reg::MaskReg maskReg = Reg::UpdateMask<T, Reg::RegTraitNumTwo>(sreg);
                    DataCopy(vSrcReg0, src0 + i * one_rep_size);
                    DataCopy(vSrcReg1, src1 + i * one_rep_size);
                    #ifdef VF
                    #if VF == 0
                        Reg::Div(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                    #elif VF == 1:
                        Reg::MulAddDst(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                    #elif VF == 2:
                        Reg::FusedAbsSub(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                    #endif
                    #endif
                    DataCopy(dst + i * one_rep_size, vDstReg0, maskReg);
                }
            }
        }

        outQueue.EnQue<T>(dstLocal);
        inQueueX.FreeTensor(src0Local);
        inQueueX2.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        DataCopy(dst0_global, dstLocal, dataSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<T> src0_global;
    GlobalTensor<SCR1_T> src1_global;
    GlobalTensor<T> dst0_global;
    GlobalTensor<T> dst1_global;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECIN, 1> inQueueX2;
    TQue<TPosition::VECOUT, 1> outQueue;
    TQue<TPosition::VECOUT, 1> outQueue2;
    uint32_t dataSize = 0;
    uint32_t mask;
};

struct MicroBinaryParams {
    void (*CallFunc)();
};

template<typename T, typename T2, int32_t mode, int16_t VF>
void MicroBinaryRunCase() {
    int byte_size = sizeof(T);
    int shape_size = 1024;
    int mask = 256;
    int dataSize = 1024;
    uint8_t dst0Gm[shape_size * byte_size] = {0};
    uint8_t dst1Gm[shape_size * byte_size] = {0};
    uint8_t src0Gm[shape_size * byte_size] = {0};
    uint8_t src1Gm[shape_size * byte_size] = {0};

    KernelBinary<T, T2, mode, VF> op;
    op.Init(dst0Gm, dst1Gm, src0Gm, src1Gm, dataSize, mask);
    op.Process();
}

class MicroBinaryB64Testsuite : public testing::Test, public testing::WithParamInterface<MicroBinaryParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(MicroBinaryB64TestCase, MicroBinaryB64Testsuite,
    ::testing::Values(
                      MicroBinaryParams { MicroBinaryRunCase<int64_t, int64_t, 0, 0> },
                      MicroBinaryParams { MicroBinaryRunCase<uint64_t, uint64_t, 0, 0> },
                      MicroBinaryParams { MicroBinaryRunCase<int64_t, int64_t, 0, 1> },
                      MicroBinaryParams { MicroBinaryRunCase<uint64_t, uint64_t, 0, 1> },
                      MicroBinaryParams { MicroBinaryRunCase<int64_t, int64_t, 0, 2> },
                      MicroBinaryParams { MicroBinaryRunCase<bfloat16_t, bfloat16_t, 0, 1> }
                      ));

TEST_P(MicroBinaryB64Testsuite, MicroBinaryB64TestCase)
{
    auto param = GetParam();
    param.CallFunc();
}
