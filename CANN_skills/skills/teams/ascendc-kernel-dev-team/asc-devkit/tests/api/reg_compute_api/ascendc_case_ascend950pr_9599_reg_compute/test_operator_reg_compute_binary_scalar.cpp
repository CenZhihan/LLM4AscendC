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
class KernelBinaryScalar {
public:
    __aicore__ inline KernelBinaryScalar()
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
        Duplicate(dst1Local, zero, dataSize);
        Duplicate(dst0Local, zero, dataSize);
        LocalTensor<T> src0Local = inQueueX.DeQue<T>();
        LocalTensor<Scr1T> src1Local = inQueueX2.DeQue<Scr1T>();
        uint16_t maskBitSize = 256;
        uint16_t oneRepSize = maskBitSize/sizeof(T);
        uint16_t rep = dataSize/oneRepSize;
        __ubuf__ T* dstPtr = (__ubuf__ T*)dst0Local.GetPhyAddr();
        __ubuf__ T* dst1Ptr = (__ubuf__ T*)dst1Local.GetPhyAddr();
        __ubuf__ T* src0Ptr = (__ubuf__ T*)src0Local.GetPhyAddr();
        __ubuf__ Scr1T* src1Ptr = (__ubuf__ Scr1T*)src1Local.GetPhyAddr();
        Scr1T scalar = src1Local.GetValue(0);
        T mulsCastScalar = src0Local.GetValue(0);
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> vSrcReg0;
            Reg::RegTensor<Scr1T> vSrcReg1;
            Reg::RegTensor<int32_t> vSrcReg2;
            Reg::RegTensor<T> vDstReg0;
            Reg::RegTensor<T> vDstReg1;
            Reg::RegTensor<half> halfReg;
            vector_bool carryOut;
            uint32_t sreg = static_cast<uint32_t>(mask);
            Reg::MaskReg maskReg;
            maskReg = Reg::UpdateMask<T>(sreg);
            for (uint16_t i = 0; i < static_cast<uint16_t>(rep); i++) {
                Reg::DataCopy(vSrcReg0, src0Ptr + i * oneRepSize);
                Reg::DataCopy(vSrcReg1, src1Ptr + i * oneRepSize);
                if constexpr (mD == 0) {
                    Reg::Adds(vDstReg1, vSrcReg0, scalar, maskReg);
                } else if constexpr (mD == 1) {
                    Reg::Muls(vDstReg1, vSrcReg0, scalar, maskReg);
                } else if constexpr (mD == 2) {
                    Reg::Maxs(vDstReg0, vSrcReg0, scalar, maskReg);
                } else if constexpr (mD == 3) {
                    Reg::Mins(vDstReg1, vSrcReg0, scalar, maskReg);
                } else if constexpr (mD == 4) {
                    Reg::ShiftLefts(vDstReg1, vSrcReg0, static_cast<int16_t>(scalar), maskReg);
                } else if constexpr (mD == 5) {
                    Reg::ShiftRights(vDstReg1, vSrcReg0, static_cast<int16_t>(scalar), maskReg);
                } else if constexpr (mD == 6) {
                    Reg::FusedMulsCast(halfReg, vSrcReg0, mulsCastScalar, maskReg);
                } else if constexpr (mD == 7) {
                    Reg::LeakyRelu(vDstReg1, vSrcReg0, scalar, maskReg);
                }
                Reg::DataCopy(dstPtr + i * oneRepSize, vDstReg0, maskReg);
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

struct MicroBinaryScalarParams {
    void (*CallFunc)();
};

template<typename T, int32_t mode>
void RunCase() {
    int fpByteSize = sizeof(T);
    int shapeSize = 1024;
    int mask = 256;
    int dataSize = 1024;
    uint8_t dst0Gm[shapeSize * fpByteSize] = {0};
    uint8_t dst1Gm[shapeSize * fpByteSize] = {0};
    uint8_t src0Gm[shapeSize * fpByteSize] = {0};
    uint8_t src1Gm[shapeSize * fpByteSize] = {0};

    KernelBinaryScalar<T, T, mode> op;
    op.Init(dst0Gm, dst1Gm, src0Gm, src1Gm, dataSize, mask);
    op.Process();
}

class MicroBinaryScalarTestsuite : public testing::Test, public testing::WithParamInterface<MicroBinaryScalarParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(MicroBinaryScalarTestCase, MicroBinaryScalarTestsuite,
    ::testing::Values(
            MicroBinaryScalarParams { RunCase<int64_t, 0> },
            MicroBinaryScalarParams { RunCase<uint64_t, 0> },
            MicroBinaryScalarParams { RunCase<uint16_t, 1> },
            MicroBinaryScalarParams { RunCase<int16_t, 1> },
            MicroBinaryScalarParams { RunCase<int32_t, 1> },
            MicroBinaryScalarParams { RunCase<int64_t, 1> },
            MicroBinaryScalarParams { RunCase<uint64_t, 1> },
            MicroBinaryScalarParams { RunCase<uint8_t, 2> },
            MicroBinaryScalarParams { RunCase<int8_t, 2> },
            MicroBinaryScalarParams { RunCase<uint16_t, 2> },
            MicroBinaryScalarParams { RunCase<uint32_t, 2> },
            MicroBinaryScalarParams { RunCase<int32_t, 2> },
            MicroBinaryScalarParams { RunCase<int64_t, 2> },
            MicroBinaryScalarParams { RunCase<uint64_t, 2> },
            MicroBinaryScalarParams { RunCase<uint8_t, 3> },
            MicroBinaryScalarParams { RunCase<int8_t, 3> },
            MicroBinaryScalarParams { RunCase<uint16_t, 3> },
            MicroBinaryScalarParams { RunCase<uint32_t, 3> },
            MicroBinaryScalarParams { RunCase<int32_t, 3> },
            MicroBinaryScalarParams { RunCase<int64_t, 3> },
            MicroBinaryScalarParams { RunCase<uint64_t, 3> },
            MicroBinaryScalarParams { RunCase<uint8_t, 4> },
            MicroBinaryScalarParams { RunCase<int8_t, 4> },
            MicroBinaryScalarParams { RunCase<uint8_t, 5> },
            MicroBinaryScalarParams { RunCase<int8_t, 5> },
            MicroBinaryScalarParams { RunCase<float, 6> },
            MicroBinaryScalarParams { RunCase<half, 7> },
            MicroBinaryScalarParams { RunCase<float, 7> }
            ));

TEST_P(MicroBinaryScalarTestsuite, MicroBinaryScalarTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}
