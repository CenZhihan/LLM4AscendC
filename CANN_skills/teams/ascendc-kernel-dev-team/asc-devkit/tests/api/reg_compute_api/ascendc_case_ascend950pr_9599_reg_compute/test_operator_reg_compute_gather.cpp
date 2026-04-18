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


#define D_TYPE half
#define D_TYPE1 half
#define MODE 8

template <typename T, typename U>
class KernelGather {
public:
    __aicore__ inline KernelGather()
    {}
    __aicore__ inline void Init(GM_ADDR dst0Gm, GM_ADDR src0Gm, GM_ADDR src1Gm,
            uint32_t nums)
    {
        src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src0Gm), nums);
        src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(src1Gm), nums);
        dst0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst0Gm), nums);

        pipe.InitBuffer(inQueueX, 1, nums * sizeof(T));
        pipe.InitBuffer(inQueueX2, 1, nums * sizeof(U));
        pipe.InitBuffer(outQueue, 1, nums * sizeof(T));
        dataSize = nums;
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
        LocalTensor<U> src1Local = inQueueX2.AllocTensor<U>();
        DataCopy(src0Local, src0Global, dataSize);
        DataCopy(src1Local, src1Global, dataSize);
        inQueueX.EnQue(src0Local);
        inQueueX2.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> dst0Local = outQueue.AllocTensor<T>();
        T zero = 0;
        Duplicate(dst0Local, zero, dataSize);
        LocalTensor<T> src0Local = inQueueX.DeQue<T>();
        LocalTensor<T> src1Local = inQueueX2.DeQue<T>();
        uint16_t maskBitSize = 256;
        uint16_t oneRepSize = maskBitSize/sizeof(T);
        uint16_t rep = dataSize/oneRepSize;
        __ubuf__ T* dstPtr = (__ubuf__ T*)dst0Local.GetPhyAddr();
        __ubuf__ T* src0Ptr = (__ubuf__ T*)src0Local.GetPhyAddr();
        __ubuf__ U* src1Ptr = (__ubuf__ U*)src1Local.GetPhyAddr();
        T scalar = src1Ptr[0];
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> vSrcReg0;
            Reg::RegTensor<U> vSrcReg1;
            Reg::RegTensor<T> vDstReg0;
            uint32_t sreg = static_cast<uint32_t>(dataSize);
            Reg::MaskReg preg;
            for (uint16_t i = 0; i < static_cast<uint16_t>(rep); i++) {
                preg = Reg::UpdateMask<T>(sreg);
                Reg::DataCopy(vSrcReg0, src0Ptr + i * oneRepSize);
                Reg::DataCopy(vSrcReg1, src1Ptr + i * oneRepSize);
                Reg::Gather<T, U>(vDstReg0, vSrcReg0, vSrcReg1);
                Reg::DataCopy(dstPtr + i * oneRepSize, vDstReg0, preg);
            }
        }
        outQueue.EnQue<T>(dst0Local);
        inQueueX.FreeTensor(src0Local);
        inQueueX2.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dst0Local = outQueue.DeQue<T>();
        DataCopy(dst0Global, dst0Local, dataSize);
        outQueue.FreeTensor(dst0Local);
    }

private:
    GlobalTensor<T> src0Global;
    GlobalTensor<U> src1Global;
    GlobalTensor<T> dst0Global;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECIN, 1> inQueueX2;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t dataSize = 0;
};

struct MicroGatherParams {
    void (*CallFunc)();
};

template<typename T, typename U>
void RunCase() {
    int byteSize = sizeof(T);
    int shapeSize = 1024;
    int dataSize = 1024;
    uint8_t dst0Gm[shapeSize * byteSize] = {0};
    uint8_t src0Gm[shapeSize * byteSize] = {0};
    uint8_t src1Gm[shapeSize * byteSize] = {0};

    KernelGather<T, U> op;
    op.Init(dst0Gm, src0Gm, src1Gm, dataSize);
    op.Process();
}

class MicroGatherTestsuite : public testing::Test, public testing::WithParamInterface<MicroGatherParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(MicroGatherTestCase, MicroGatherTestsuite,
    ::testing::Values(MicroGatherParams { RunCase<uint16_t, uint16_t> },
                      MicroGatherParams { RunCase<int16_t, uint16_t> },
                      MicroGatherParams { RunCase<half, uint16_t> },
                      MicroGatherParams { RunCase<bfloat16_t, uint16_t> },
                      MicroGatherParams { RunCase<uint32_t, uint32_t> },
                      MicroGatherParams { RunCase<int32_t, uint32_t> },
                      MicroGatherParams { RunCase<float, uint32_t> },
                      MicroGatherParams { RunCase<int8_t, uint8_t> },
                      MicroGatherParams { RunCase<uint8_t, uint8_t> }
                      ));

TEST_P(MicroGatherTestsuite, MicroGatherTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}
