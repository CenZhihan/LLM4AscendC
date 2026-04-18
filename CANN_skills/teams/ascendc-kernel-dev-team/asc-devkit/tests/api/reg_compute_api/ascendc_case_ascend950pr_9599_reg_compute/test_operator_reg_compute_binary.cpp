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

template <typename T, typename Scr1T, int32_t mD>
class KernelBinary {
public:
    __aicore__ inline KernelBinary()
    {}
    __aicore__ inline void Init(GM_ADDR dst0Gm, GM_ADDR dst1Gm, GM_ADDR src0Gm, GM_ADDR src1Gm,
            uint32_t nums, uint32_t vecMask)
    {
        src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src0Gm), nums);
        src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ Scr1T *>(src1Gm), nums);
        dst0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst0Gm), nums);
        dst1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst1Gm), nums);

        pipe.InitBuffer(inQueueX, 1, nums * sizeof(T));
        pipe.InitBuffer(inQueueX2, 1, nums * sizeof(Scr1T));
        pipe.InitBuffer(outQueue, 1, nums * sizeof(T));
        pipe.InitBuffer(outQueue2, 1, nums * sizeof(T));
        dataSize = nums;
        mask = vecMask;
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
        LocalTensor<Scr1T> src1Local = inQueueX2.AllocTensor<Scr1T>();
        DataCopy(src0Local, src0Global, dataSize);
        DataCopy(src1Local, src1Global, dataSize);
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
            Duplicate(tmp0, static_cast<int8_t>(0), dataSize);
            Duplicate(tmp1, static_cast<int8_t>(0), dataSize);
        } else {
            Duplicate(dst1Local, zero, dataSize);
            Duplicate(dst0Local, zero, dataSize);
        }
        LocalTensor<T> src0Local = inQueueX.DeQue<T>();
        LocalTensor<Scr1T> src1Local = inQueueX2.DeQue<Scr1T>();
        static constexpr Reg::DivSpecificMode divMode = {Reg::MaskMergeMode::ZEROING, true};
        uint16_t maskBitSize = 256;
        uint16_t oneRepSize = maskBitSize/sizeof(T);
        uint16_t rep = dataSize/oneRepSize;
        __ubuf__ T* dstPtr = (__ubuf__ T*)dst0Local.GetPhyAddr();
        __ubuf__ T* dst1Ptr = (__ubuf__ T*)dst1Local.GetPhyAddr();
        __ubuf__ T* src0Ptr = (__ubuf__ T*)src0Local.GetPhyAddr();
        __ubuf__ Scr1T* src1Ptr = (__ubuf__ Scr1T*)src1Local.GetPhyAddr();
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> vSrcReg0;
            Reg::RegTensor<Scr1T> vSrcReg1;
            Reg::RegTensor<int32_t> vSrcReg2;
            Reg::RegTensor<T> vDstReg0;
            Reg::RegTensor<T> vDstReg1;
            Reg::RegTensor<float> fp32DstReg;
            vector_bool carryOut;
            vector_bool carrySrc;
            uint32_t sreg = static_cast<uint32_t>(mask);
            Reg::MaskReg maskReg;
            maskReg = Reg::UpdateMask<T>(sreg);
            for (uint16_t i = 0; i < static_cast<uint16_t>(rep); i++) {
                Reg::DataCopy(vSrcReg0, src0Ptr + i * oneRepSize);
                Reg::DataCopy(vSrcReg1, src1Ptr + i * oneRepSize);
                if constexpr (mD == 0) {
                    Reg::Add(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 1) {
                    Reg::Sub(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 2) {
                    Reg::Mul(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 3) {
                    Reg::Div(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 4) {
                    Reg::Max(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 5) {
                    Reg::Min(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 6) {
                    Reg::ShiftLeft(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 7) {
                    Reg::ShiftRight(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 8) {
                    Reg::And(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 9) {
                    Reg::Or(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 10) {
                    Reg::Xor(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 11) {
                    Reg::Mull(vDstReg0, vDstReg1, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 12) {
                    carryOut = pset_b8(PAT_ALL);
                    vaddc(carryOut, vDstReg1, vSrcReg0, vSrcReg1, maskReg);
                    psts(carryOut, (__ubuf__ uint32_t *)dst1Ptr, i * oneRepSize, NORM);
                } else if constexpr (mD == 13) {
                    carryOut = pset_b8(PAT_ALL);
                    vsubc(carryOut, vDstReg1, vSrcReg0, vSrcReg1, maskReg);
                    psts(carryOut, (__ubuf__ uint32_t *)dst1Ptr, i * oneRepSize, NORM);
                } else if constexpr (mD == 14) {
                    carryOut = pset_b8(PAT_ALL);
                    vsubcs(carryOut, vDstReg1, vSrcReg0, vSrcReg1, carrySrc, maskReg);
                    psts(carryOut, (__ubuf__ uint32_t *)dst1Ptr, i * oneRepSize, NORM);
                } else if constexpr (mD == 17) {
                    Reg::Div<T, &divMode>(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 20) {
                    Reg::FusedExpSub(fp32DstReg, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 21) {
                    Reg::FusedAbsSub(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 22) {
                    Prelu(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                } else if constexpr (mD == 23) {
                    Reg::FusedMulDstAdd(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                }
                Reg::DataCopy(dstPtr + i * oneRepSize, vDstReg0, maskReg);
                if constexpr (mD == 15) {
                    Reg::DataCopy(dst1Ptr + i * oneRepSize, vDstReg1, maskReg);
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
        DataCopy(dst0Global, dst0Local, dataSize);
        DataCopy(dst1Global, dst1Local, dataSize);
        outQueue.FreeTensor(dst0Local);
        outQueue2.FreeTensor(dst1Local);
    }

private:
    GlobalTensor<T> src0Global;
    GlobalTensor<Scr1T> src1Global;
    GlobalTensor<T> dst0Global;
    GlobalTensor<T> dst1Global;

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

template<typename T, typename T2, int32_t mode>
void MicroBinaryRunCase() {
    int byteSize = sizeof(T);
    int shapeSize = 1024;
    int mask = 256;
    int dataSize = 1024;
    uint8_t dst0Gm[shapeSize * byteSize] = {0};
    uint8_t dst1Gm[shapeSize * byteSize] = {0};
    uint8_t src0Gm[shapeSize * byteSize] = {0};
    uint8_t src1Gm[shapeSize * byteSize] = {0};

    KernelBinary<T, T2, mode> op;
    op.Init(dst0Gm, dst1Gm, src0Gm, src1Gm, dataSize, mask);
    op.Process();
}

class MicroBinaryTestsuite : public testing::Test, public testing::WithParamInterface<MicroBinaryParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(MicroBinaryTestCase, MicroBinaryTestsuite,
    ::testing::Values(
                      MicroBinaryParams { MicroBinaryRunCase<uint8_t, uint8_t, 0> },
                      MicroBinaryParams { MicroBinaryRunCase<int8_t, int8_t, 0> },
                      MicroBinaryParams { MicroBinaryRunCase<int64_t, int64_t, 0> },
                      MicroBinaryParams { MicroBinaryRunCase<uint64_t, uint64_t, 0> },
                      MicroBinaryParams { MicroBinaryRunCase<int32_t, int32_t, 1> },
                      MicroBinaryParams { MicroBinaryRunCase<uint32_t, uint32_t, 1> },
                      MicroBinaryParams { MicroBinaryRunCase<uint16_t, uint16_t, 1> },
                      MicroBinaryParams { MicroBinaryRunCase<int16_t, int16_t, 1> },
                      MicroBinaryParams { MicroBinaryRunCase<uint8_t, uint8_t, 1> },
                      MicroBinaryParams { MicroBinaryRunCase<int8_t, int8_t, 1> },
                      MicroBinaryParams { MicroBinaryRunCase<int64_t, int64_t, 2> },
                      MicroBinaryParams { MicroBinaryRunCase<uint64_t, uint64_t, 2> },
                      MicroBinaryParams { MicroBinaryRunCase<uint32_t, uint32_t, 2> },
                      MicroBinaryParams { MicroBinaryRunCase<int32_t, int32_t, 2> },
                      MicroBinaryParams { MicroBinaryRunCase<half, half, 3> },
                      MicroBinaryParams { MicroBinaryRunCase<float, float, 3> },
                      MicroBinaryParams { MicroBinaryRunCase<int64_t, int64_t, 8> },
                      MicroBinaryParams { MicroBinaryRunCase<uint64_t, uint64_t, 8> },
                      MicroBinaryParams { MicroBinaryRunCase<int32_t, int32_t, 8> },
                      MicroBinaryParams { MicroBinaryRunCase<uint32_t, uint32_t, 8> },
                      MicroBinaryParams { MicroBinaryRunCase<uint16_t, uint16_t, 8> },
                      MicroBinaryParams { MicroBinaryRunCase<int16_t, int16_t, 8> },
                      MicroBinaryParams { MicroBinaryRunCase<uint8_t, uint8_t, 8> },
                      MicroBinaryParams { MicroBinaryRunCase<bool, bool, 8> },
                      MicroBinaryParams { MicroBinaryRunCase<int8_t, int8_t, 8> },
                      MicroBinaryParams { MicroBinaryRunCase<int64_t, int64_t, 4> },
                      MicroBinaryParams { MicroBinaryRunCase<uint64_t, uint64_t, 4> },
                      MicroBinaryParams { MicroBinaryRunCase<int32_t, int32_t, 4> },
                      MicroBinaryParams { MicroBinaryRunCase<uint32_t, uint32_t, 4> },
                      MicroBinaryParams { MicroBinaryRunCase<uint16_t, uint16_t, 4> },
                      MicroBinaryParams { MicroBinaryRunCase<int16_t, int16_t, 4> },
                      MicroBinaryParams { MicroBinaryRunCase<uint8_t, uint8_t, 4> },
                      MicroBinaryParams { MicroBinaryRunCase<int8_t, int8_t, 4> },
                      MicroBinaryParams { MicroBinaryRunCase<int64_t, int64_t, 5> },
                      MicroBinaryParams { MicroBinaryRunCase<uint64_t, uint64_t, 5> },
                      MicroBinaryParams { MicroBinaryRunCase<int32_t, int32_t, 5> },
                      MicroBinaryParams { MicroBinaryRunCase<uint32_t, uint32_t, 5> },
                      MicroBinaryParams { MicroBinaryRunCase<uint16_t, uint16_t, 5> },
                      MicroBinaryParams { MicroBinaryRunCase<int16_t, int16_t, 5> },
                      MicroBinaryParams { MicroBinaryRunCase<uint8_t, uint8_t, 5> },
                      MicroBinaryParams { MicroBinaryRunCase<int8_t, int8_t, 5> },
                      MicroBinaryParams { MicroBinaryRunCase<int64_t, int64_t, 9> },
                      MicroBinaryParams { MicroBinaryRunCase<uint64_t, uint64_t, 9> },
                      MicroBinaryParams { MicroBinaryRunCase<int32_t, int32_t, 9> },
                      MicroBinaryParams { MicroBinaryRunCase<uint32_t, uint32_t, 9> },
                      MicroBinaryParams { MicroBinaryRunCase<uint16_t, uint16_t, 9> },
                      MicroBinaryParams { MicroBinaryRunCase<int16_t, int16_t, 9> },
                      MicroBinaryParams { MicroBinaryRunCase<uint8_t, uint8_t, 9> },
                      MicroBinaryParams { MicroBinaryRunCase<int8_t, int8_t, 9> },
                      MicroBinaryParams { MicroBinaryRunCase<bool, bool, 9> },
                      MicroBinaryParams { MicroBinaryRunCase<int64_t, int64_t, 10> },
                      MicroBinaryParams { MicroBinaryRunCase<uint64_t, uint64_t, 10> },
                      MicroBinaryParams { MicroBinaryRunCase<int8_t, int8_t, 10> },
                      MicroBinaryParams { MicroBinaryRunCase<uint8_t, uint8_t, 10> },
                      MicroBinaryParams { MicroBinaryRunCase<bool, bool, 10> },
                      MicroBinaryParams { MicroBinaryRunCase<int16_t, int16_t, 10> },
                      MicroBinaryParams { MicroBinaryRunCase<uint16_t, uint16_t, 10> },
                      MicroBinaryParams { MicroBinaryRunCase<uint32_t, uint32_t, 11> },
                      MicroBinaryParams { MicroBinaryRunCase<int32_t, int32_t, 11> },
                      MicroBinaryParams { MicroBinaryRunCase<int64_t, int64_t, 6> },
                      MicroBinaryParams { MicroBinaryRunCase<uint64_t, int64_t, 6> },
                      MicroBinaryParams { MicroBinaryRunCase<int32_t, int32_t, 6> },
                      MicroBinaryParams { MicroBinaryRunCase<uint32_t, int32_t, 6> },
                      MicroBinaryParams { MicroBinaryRunCase<uint16_t, int16_t, 6> },
                      MicroBinaryParams { MicroBinaryRunCase<int16_t, int16_t, 6> },
                      MicroBinaryParams { MicroBinaryRunCase<uint8_t, int8_t, 6> },
                      MicroBinaryParams { MicroBinaryRunCase<int8_t, int8_t, 6> },
                      MicroBinaryParams { MicroBinaryRunCase<int64_t, int64_t, 7> },
                      MicroBinaryParams { MicroBinaryRunCase<uint64_t, int64_t, 7> },
                      MicroBinaryParams { MicroBinaryRunCase<int32_t, int32_t, 7> },
                      MicroBinaryParams { MicroBinaryRunCase<uint32_t, int32_t, 7> },
                      MicroBinaryParams { MicroBinaryRunCase<uint16_t, int16_t, 7> },
                      MicroBinaryParams { MicroBinaryRunCase<int16_t, int16_t, 7> },
                      MicroBinaryParams { MicroBinaryRunCase<uint8_t, int8_t, 7> },
                      MicroBinaryParams { MicroBinaryRunCase<int8_t, int8_t, 7> },
                      MicroBinaryParams { MicroBinaryRunCase<uint32_t, uint32_t, 13> },
                      MicroBinaryParams { MicroBinaryRunCase<uint32_t, uint32_t, 14> },
                      MicroBinaryParams { MicroBinaryRunCase<int32_t, int32_t, 13> },
                      MicroBinaryParams { MicroBinaryRunCase<int32_t, int32_t, 14> },
                      MicroBinaryParams { MicroBinaryRunCase<float, float, 22> },
                      MicroBinaryParams { MicroBinaryRunCase<half, half, 22> },
                      MicroBinaryParams { MicroBinaryRunCase<int32_t, int32_t, 15> },
                      MicroBinaryParams { MicroBinaryRunCase<int32_t, int32_t, 16> },
                      MicroBinaryParams { MicroBinaryRunCase<float, float, 17> },
                      MicroBinaryParams { MicroBinaryRunCase<int32_t, int32_t, 19> },
                      MicroBinaryParams { MicroBinaryRunCase<uint32_t, uint32_t, 19> },
                      MicroBinaryParams { MicroBinaryRunCase<int16_t, int16_t, 19> },
                      MicroBinaryParams { MicroBinaryRunCase<uint16_t, uint16_t, 19> },
                      MicroBinaryParams { MicroBinaryRunCase<float, float, 19> },
                      MicroBinaryParams { MicroBinaryRunCase<half, half, 19> },
                      MicroBinaryParams { MicroBinaryRunCase<bfloat16_t, bfloat16_t, 19> },
                      MicroBinaryParams { MicroBinaryRunCase<float, float, 20> },
                      MicroBinaryParams { MicroBinaryRunCase<half, half, 20> },
                      MicroBinaryParams { MicroBinaryRunCase<float, float, 21> },
                      MicroBinaryParams { MicroBinaryRunCase<half, half, 21> },
                      MicroBinaryParams { MicroBinaryRunCase<half, half, 23> },
                      MicroBinaryParams { MicroBinaryRunCase<float, float, 23> }
                      ));

TEST_P(MicroBinaryTestsuite, MicroBinaryTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}
