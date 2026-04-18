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
class KernelCmp {
public:
    __aicore__ inline KernelCmp()
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
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> vSrcReg0;
            Reg::RegTensor<SCR1_T> vSrcReg1;
            Reg::RegTensor<T> vDstReg0;
            Reg::RegTensor<T> vDstReg1;
            vector_bool carryOut;
            vector_bool cmpReg;
            uint32_t sreg = (uint32_t)mask;
            Reg::MaskReg maskReg;
            maskReg = Reg::UpdateMask<T>(sreg);
            for (uint16_t i = 0; i < (uint16_t)rep; i++) {
                Reg::DataCopy(vSrcReg0, src0Ptr + i * one_rep_size);
                Reg::DataCopy(vSrcReg1, src1Ptr + i * one_rep_size);
                if constexpr (MD == 0) {
                    Reg::Compare<T, CMPMODE::EQ>(cmpReg, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (MD == 1) {
                    Reg::Compare<T, CMPMODE::GE>(cmpReg, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (MD == 2) {
                    Reg::Compare<T, CMPMODE::GT>(cmpReg, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (MD == 3) {
                    Reg::Compare<T, CMPMODE::LE>(cmpReg, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (MD == 4) {
                    Reg::Compare<T, CMPMODE::LT>(cmpReg, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (MD == 5) {
                    Reg::Compare<T, CMPMODE::NE>(cmpReg, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (MD == 6) {
                    Reg::CompareScalar<T, CMPMODE::EQ>(cmpReg, vSrcReg0, scalar, maskReg);
                } else if constexpr (MD == 7) {
                    Reg::CompareScalar<T, CMPMODE::GE>(cmpReg, vSrcReg0, scalar, maskReg);
                } else if constexpr (MD == 8) {
                    Reg::CompareScalar<T, CMPMODE::GT>(cmpReg, vSrcReg0, scalar, maskReg);
                } else if constexpr (MD == 9) {
                    Reg::CompareScalar<T, CMPMODE::LE>(cmpReg, vSrcReg0, scalar, maskReg);
                } else if constexpr (MD == 10) {
                    Reg::CompareScalar<T, CMPMODE::LT>(cmpReg, vSrcReg0, scalar, maskReg);
                } else if constexpr (MD == 11) {
                    Reg::CompareScalar<T, CMPMODE::NE>(cmpReg, vSrcReg0, scalar, maskReg);
                } else if constexpr (MD == 12) {
                    if constexpr (std::is_same<T, bool>::value) {
                        cmpReg = Reg::CreateMask<int8_t, Reg::MaskPattern::ALLF>();
                        Reg::Select(vDstReg1, vSrcReg0, vSrcReg1, cmpReg);
                    } else {
                        Reg::Compare<T, CMPMODE::EQ>(cmpReg, vSrcReg0, vSrcReg1, maskReg);
                        Reg::Select(vDstReg1, vSrcReg0, vSrcReg1, cmpReg);
                    }
                }
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

struct MicroCmpParams {
    void (*CallFunc)();
};

template<typename T, int32_t mode>
void RunCase() {
    int byte_size = sizeof(T);
    int shape_size = 1024;
    int mask = 256;
    int dataSize = 1024;
    uint8_t dst0Gm[shape_size * byte_size] = {0};
    uint8_t dst1Gm[shape_size * byte_size] = {0};
    uint8_t src0Gm[shape_size * byte_size] = {0};
    uint8_t src1Gm[shape_size * byte_size] = {0};

    KernelCmp<T, T, mode> op;
    op.Init(dst0Gm, dst1Gm, src0Gm, src1Gm, dataSize, mask);
    op.Process();
}

class MicroCmpTestsuite : public testing::Test, public testing::WithParamInterface<MicroCmpParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(MicroCmpTestCase, MicroCmpTestsuite,
    ::testing::Values(MicroCmpParams { RunCase<uint16_t, 0> },
                      MicroCmpParams { RunCase<int16_t, 0> },
                      MicroCmpParams { RunCase<uint32_t, 0> },
                      MicroCmpParams { RunCase<int32_t, 0> },
                      MicroCmpParams { RunCase<uint64_t, 0> },
                      MicroCmpParams { RunCase<int64_t, 0> },
                      MicroCmpParams { RunCase<uint16_t, 1> },
                      MicroCmpParams { RunCase<int16_t, 1> },
                      MicroCmpParams { RunCase<uint32_t, 1> },
                      MicroCmpParams { RunCase<int32_t, 1> },
                      MicroCmpParams { RunCase<uint64_t, 1> },
                      MicroCmpParams { RunCase<int64_t, 1> },
                      MicroCmpParams { RunCase<uint16_t, 2> },
                      MicroCmpParams { RunCase<int16_t, 2> },
                      MicroCmpParams { RunCase<uint32_t, 2> },
                      MicroCmpParams { RunCase<int32_t, 2> },
                      MicroCmpParams { RunCase<uint64_t, 2> },
                      MicroCmpParams { RunCase<int64_t, 2> },
                      MicroCmpParams { RunCase<uint16_t, 3> },
                      MicroCmpParams { RunCase<int16_t, 3> },
                      MicroCmpParams { RunCase<uint32_t, 3> },
                      MicroCmpParams { RunCase<int32_t, 3> },
                      MicroCmpParams { RunCase<uint64_t, 3> },
                      MicroCmpParams { RunCase<int64_t, 3> },
                      MicroCmpParams { RunCase<uint16_t, 4> },
                      MicroCmpParams { RunCase<int16_t, 4> },
                      MicroCmpParams { RunCase<uint32_t, 4> },
                      MicroCmpParams { RunCase<int32_t, 4> },
                      MicroCmpParams { RunCase<uint64_t, 4> },
                      MicroCmpParams { RunCase<int64_t, 4> },
                      MicroCmpParams { RunCase<uint16_t, 5> },
                      MicroCmpParams { RunCase<int16_t, 5> },
                      MicroCmpParams { RunCase<uint32_t, 5> },
                      MicroCmpParams { RunCase<int32_t, 5> },
                      MicroCmpParams { RunCase<uint64_t, 5> },
                      MicroCmpParams { RunCase<int64_t, 5> },
                      MicroCmpParams { RunCase<uint8_t, 6> },
                      MicroCmpParams { RunCase<int8_t, 6> },
                      MicroCmpParams { RunCase<uint16_t, 6> },
                      MicroCmpParams { RunCase<int16_t, 6> },
                      MicroCmpParams { RunCase<uint32_t, 6> },
                      MicroCmpParams { RunCase<int32_t, 6> },
                      MicroCmpParams { RunCase<uint64_t, 6> },
                      MicroCmpParams { RunCase<int64_t, 6> },
                      MicroCmpParams { RunCase<uint8_t, 7> },
                      MicroCmpParams { RunCase<int8_t, 7> },
                      MicroCmpParams { RunCase<uint16_t, 7> },
                      MicroCmpParams { RunCase<int16_t, 7> },
                      MicroCmpParams { RunCase<uint32_t, 7> },
                      MicroCmpParams { RunCase<int32_t, 7> },
                      MicroCmpParams { RunCase<uint64_t, 7> },
                      MicroCmpParams { RunCase<int64_t, 7> },
                      MicroCmpParams { RunCase<uint8_t, 8> },
                      MicroCmpParams { RunCase<int8_t, 8> },
                      MicroCmpParams { RunCase<uint16_t, 8> },
                      MicroCmpParams { RunCase<int16_t, 8> },
                      MicroCmpParams { RunCase<uint32_t, 8> },
                      MicroCmpParams { RunCase<int32_t, 8> },
                      MicroCmpParams { RunCase<uint64_t, 8> },
                      MicroCmpParams { RunCase<int64_t, 8> },
                      MicroCmpParams { RunCase<uint8_t, 9> },
                      MicroCmpParams { RunCase<int8_t, 9> },
                      MicroCmpParams { RunCase<uint16_t, 9> },
                      MicroCmpParams { RunCase<int16_t, 9> },
                      MicroCmpParams { RunCase<uint32_t, 9> },
                      MicroCmpParams { RunCase<int32_t, 9> },
                      MicroCmpParams { RunCase<uint64_t, 9> },
                      MicroCmpParams { RunCase<int64_t, 9> },
                      MicroCmpParams { RunCase<uint8_t, 10> },
                      MicroCmpParams { RunCase<int8_t, 10> },
                      MicroCmpParams { RunCase<uint16_t, 10> },
                      MicroCmpParams { RunCase<int16_t, 10> },
                      MicroCmpParams { RunCase<uint32_t, 10> },
                      MicroCmpParams { RunCase<int32_t, 10> },
                      MicroCmpParams { RunCase<uint64_t, 10> },
                      MicroCmpParams { RunCase<int64_t, 10> },
                      MicroCmpParams { RunCase<uint8_t, 11> },
                      MicroCmpParams { RunCase<int8_t, 11> },
                      MicroCmpParams { RunCase<uint16_t, 11> },
                      MicroCmpParams { RunCase<int16_t, 11> },
                      MicroCmpParams { RunCase<uint32_t, 11> },
                      MicroCmpParams { RunCase<int32_t, 11> },
                      MicroCmpParams { RunCase<uint64_t, 11> },
                      MicroCmpParams { RunCase<int64_t, 11> },
                      MicroCmpParams { RunCase<bool, 12> },
                      MicroCmpParams { RunCase<uint8_t, 12> },
                      MicroCmpParams { RunCase<int8_t, 12> },
                      MicroCmpParams { RunCase<uint16_t, 12> },
                      MicroCmpParams { RunCase<int16_t, 12> },
                      MicroCmpParams { RunCase<uint32_t, 12> },
                      MicroCmpParams { RunCase<int32_t, 12> },
                      MicroCmpParams { RunCase<uint64_t, 12> },
                      MicroCmpParams { RunCase<int64_t, 12> }
                      ));

TEST_P(MicroCmpTestsuite, MicroCmpTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}
