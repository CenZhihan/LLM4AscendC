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

template <typename T, typename U>
class KernelGather {
public:
    __aicore__ inline KernelGather()
    {}
    __aicore__ inline void Init(GM_ADDR dst0_gm, GM_ADDR src0_gm, GM_ADDR src1_gm,
            uint32_t nums)
    {
        src0_global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src0_gm), nums);
        src1_global.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(src1_gm), nums);
        dst0_global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst0_gm), nums);

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
        DataCopy(src0Local, src0_global, dataSize);
        DataCopy(src1Local, src1_global, dataSize);
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
        uint16_t mask_bit_size = 256;
        uint16_t one_rep_size = mask_bit_size/sizeof(T);
        uint16_t rep = dataSize/one_rep_size;
        __ubuf__ T* dstPtr = (__ubuf__ T*)dst0Local.GetPhyAddr();
        __ubuf__ T* src0Ptr = (__ubuf__ T*)src0Local.GetPhyAddr();
        __ubuf__ U* src1Ptr = (__ubuf__ U*)src1Local.GetPhyAddr();
        T scalar = src1Ptr[0];
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> vSrcReg0;
            Reg::RegTensor<U> vSrcReg1;
            Reg::RegTensor<T> vDstReg0;
            uint32_t sreg = (uint32_t)dataSize;
            Reg::MaskReg preg;
            for (uint16_t i = 0; i < (uint16_t)rep; i++) {
                preg = Reg::UpdateMask<T>(sreg);
                Reg::DataCopy(vSrcReg0, src0Ptr + i * one_rep_size);
                Reg::DataCopy(vSrcReg1, src1Ptr + i * one_rep_size);
                Reg::Gather<T, U>(vDstReg0, vSrcReg0, vSrcReg1);
                Reg::DataCopy(dstPtr + i * one_rep_size, vDstReg0, preg);
            }
        }
        outQueue.EnQue<T>(dst0Local);
        inQueueX.FreeTensor(src0Local);
        inQueueX2.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dst0Local = outQueue.DeQue<T>();
        DataCopy(dst0_global, dst0Local, dataSize);
        outQueue.FreeTensor(dst0Local);
    }

private:
    GlobalTensor<T> src0_global;
    GlobalTensor<U> src1_global;
    GlobalTensor<T> dst0_global;

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
    int byte_size = sizeof(T);
    int shape_size = 1024;
    int dataSize = 1024;
    uint8_t dst0Gm[shape_size * byte_size] = {0};
    uint8_t src0Gm[shape_size * byte_size] = {0};
    uint8_t src1Gm[shape_size * byte_size] = {0};

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
