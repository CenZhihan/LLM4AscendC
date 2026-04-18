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

template <typename T, typename SCR1_T, int32_t MD>
class KernelUnary {
public:
    __aicore__ inline KernelUnary()
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
        LocalTensor<T> dst0Local = outQueue.AllocTensor<T>();
        LocalTensor<T> dst1Local = outQueue2.AllocTensor<T>();
        T zero = 0;
        if constexpr (std::is_same_v<T, bool>) {
            LocalTensor<int8_t> tmp0 = dst0Local.template ReinterpretCast<int8_t>();
            LocalTensor<int8_t> tmp1 = dst1Local.template ReinterpretCast<int8_t>();
            Duplicate(tmp0, (int8_t)0, dataSize);
            Duplicate(tmp1, (int8_t)0, dataSize);
        } else {
            Duplicate(dst1Local, zero, dataSize);
            Duplicate(dst0Local, zero, dataSize);
        }
        LocalTensor<T> src0Local = inQueueX.DeQue<T>();
        LocalTensor<SCR1_T> src1Local = inQueueX2.DeQue<SCR1_T>();
        uint16_t mask_bit_size = 256;
        uint16_t one_rep_size = mask_bit_size/sizeof(T);
        uint16_t rep = dataSize/one_rep_size;
        __ubuf__ T* dstPtr = (__ubuf__ T*)dst0Local.GetPhyAddr();
        __ubuf__ T* dst1Ptr = (__ubuf__ T*)dst1Local.GetPhyAddr();
        __ubuf__ T* src0Ptr = (__ubuf__ T*)src0Local.GetPhyAddr();
        __ubuf__ SCR1_T* src1Ptr = (__ubuf__ SCR1_T*)src1Local.GetPhyAddr();
        T scalar = 1;
        static constexpr Reg::SqrtSpecificMode sqrtMode = {Reg::MaskMergeMode::ZEROING, true};
        __VEC_SCOPE__
        {
            using Reg::RegTensor;
            RegTensor<T> vSrcReg0;
            RegTensor<SCR1_T> vSrcReg1;
            RegTensor<T> vDstReg0;
            RegTensor<T> vDstReg1;
            vector_bool carryOut;
            vector_bool cmpReg;
            uint32_t sreg = (uint32_t)mask;
            Reg::MaskReg maskReg;
            maskReg = Reg::UpdateMask<T>(sreg);
            for (uint16_t i = 0; i < (uint16_t)rep; i++) {
                Reg::DataCopy(vSrcReg0, src0Ptr + i * one_rep_size);
                Reg::DataCopy(vSrcReg1, src1Ptr + i * one_rep_size);
                if constexpr (MD == 0) {
                    Reg::Abs(vDstReg1, vSrcReg0, maskReg);
                } else if constexpr (MD == 1) {
                    Reg::Relu(vDstReg1, vSrcReg0, maskReg);
                } else if constexpr (MD == 2) {
                    Reg::Exp(vDstReg1, vSrcReg0, maskReg);
                } else if constexpr (MD == 3) {
                    Reg::Sqrt(vDstReg1, vSrcReg0, maskReg);
                    if constexpr (std::is_same<T, float>::value) {
                        Reg::Sqrt<T, &sqrtMode>(vDstReg1, vSrcReg0, maskReg);
                    }
                } else if constexpr (MD == 5) {
                    Reg::Log(vDstReg1, vSrcReg0, maskReg);
                } else if constexpr (MD == 6) {
                    Reg::Neg(vDstReg1, vSrcReg0, maskReg);
                } else if constexpr (MD == 7) {
                    Reg::Not(vDstReg1, vSrcReg0, maskReg);
                    Not(vDstReg1, vSrcReg0, maskReg);
                } else if constexpr (MD == 8) {
                    Reg::Log2(vDstReg1, vSrcReg0, maskReg);
                } else if constexpr (MD == 9) {
                    Reg::Log10(vDstReg1, vSrcReg0, maskReg);
                } else if constexpr (MD == 10) {
                    Reg::Copy(vDstReg1, vSrcReg0, maskReg);
                    Reg::Copy(vDstReg1, vSrcReg0);
                }
                DataCopy(dstPtr + i * one_rep_size, vDstReg1, maskReg);
            }
        }
        outQueue.EnQue<T>(dst0Local);
        outQueue2.EnQue<T>(dst1Local);
        inQueueX.FreeTensor(src0Local);
        inQueueX2.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dst0Local = outQueue.DeQue<T>();
        LocalTensor<T> dst1Local = outQueue2.DeQue<T>();
        DataCopy(dst0_global, dst0Local, dataSize);
        DataCopy(dst1_global, dst1Local, dataSize);
        outQueue.FreeTensor(dst0Local);
        outQueue2.FreeTensor(dst1Local);
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

struct MicroUnaryParams {
    void (*CallFunc)();
};

template<typename T, int32_t mode>
void MicroUnaryRunCase() {
    int byte_size = sizeof(T);
    int shape_size = 1024;
    int mask = 256;
    int dataSize = 1024;
    uint8_t dst0Gm[shape_size * byte_size] = {0};
    uint8_t dst1Gm[shape_size * byte_size] = {0};
    uint8_t src0Gm[shape_size * byte_size] = {0};
    uint8_t src1Gm[shape_size * byte_size] = {0};

    KernelUnary<T, T, mode> op;
    op.Init(dst0Gm, dst1Gm, src0Gm, src1Gm, dataSize, mask);
    op.Process();
}

class MicroUnaryTestsuite : public testing::Test, public testing::WithParamInterface<MicroUnaryParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(MicroUnaryTestCase, MicroUnaryTestsuite,
    ::testing::Values(
                    MicroUnaryParams { MicroUnaryRunCase<half, 0> },
                    MicroUnaryParams { MicroUnaryRunCase<int64_t, 0> },
                    MicroUnaryParams { MicroUnaryRunCase<int32_t, 0> },
                    MicroUnaryParams { MicroUnaryRunCase<int8_t, 0> },
                    MicroUnaryParams { MicroUnaryRunCase<half, 1> },
                    MicroUnaryParams { MicroUnaryRunCase<float, 1> },
                    MicroUnaryParams { MicroUnaryRunCase<int32_t, 1> },
                    MicroUnaryParams { MicroUnaryRunCase<int64_t, 1> },
                    MicroUnaryParams { MicroUnaryRunCase<half, 2> },
                    MicroUnaryParams { MicroUnaryRunCase<float, 2> },
                    MicroUnaryParams { MicroUnaryRunCase<half, 3> },
                    MicroUnaryParams { MicroUnaryRunCase<float, 3> },
                    MicroUnaryParams { MicroUnaryRunCase<half, 5> },
                    MicroUnaryParams { MicroUnaryRunCase<float, 5> },
                    MicroUnaryParams { MicroUnaryRunCase<int8_t, 6> },
                    MicroUnaryParams { MicroUnaryRunCase<int16_t, 6> },
                    MicroUnaryParams { MicroUnaryRunCase<int32_t, 6> },
                    MicroUnaryParams { MicroUnaryRunCase<int64_t, 6> },
                    MicroUnaryParams { MicroUnaryRunCase<half, 6> },
                    MicroUnaryParams { MicroUnaryRunCase<float, 6> },
                    MicroUnaryParams { MicroUnaryRunCase<uint8_t, 7> },
                    MicroUnaryParams { MicroUnaryRunCase<int8_t, 7> },
                    MicroUnaryParams { MicroUnaryRunCase<uint16_t, 7> },
                    MicroUnaryParams { MicroUnaryRunCase<int16_t, 7> },
                    MicroUnaryParams { MicroUnaryRunCase<uint32_t, 7> },
                    MicroUnaryParams { MicroUnaryRunCase<int32_t, 7> },
                    MicroUnaryParams { MicroUnaryRunCase<uint64_t, 7> },
                    MicroUnaryParams { MicroUnaryRunCase<int64_t, 7> },
                    MicroUnaryParams { MicroUnaryRunCase<half, 7> },
                    MicroUnaryParams { MicroUnaryRunCase<float, 7> },
                    MicroUnaryParams { MicroUnaryRunCase<half, 8> },
                    MicroUnaryParams { MicroUnaryRunCase<float, 8> },
                    MicroUnaryParams { MicroUnaryRunCase<half, 9> },
                    MicroUnaryParams { MicroUnaryRunCase<float, 9> },
                    MicroUnaryParams { MicroUnaryRunCase<bool, 10> },
                    MicroUnaryParams { MicroUnaryRunCase<uint8_t, 10> },
                    MicroUnaryParams { MicroUnaryRunCase<int8_t, 10> },
                    MicroUnaryParams { MicroUnaryRunCase<uint16_t, 10> },
                    MicroUnaryParams { MicroUnaryRunCase<int16_t, 10> },
                    MicroUnaryParams { MicroUnaryRunCase<uint32_t, 10> },
                    MicroUnaryParams { MicroUnaryRunCase<int32_t, 10> },
                    MicroUnaryParams { MicroUnaryRunCase<uint64_t, 10> },
                    MicroUnaryParams { MicroUnaryRunCase<int64_t, 10> },
                    MicroUnaryParams { MicroUnaryRunCase<half, 10> },
                    MicroUnaryParams { MicroUnaryRunCase<float, 10> }
                    ));

TEST_P(MicroUnaryTestsuite, MicroUnaryTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}
