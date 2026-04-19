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

#include <mockcpp/mockcpp.hpp>

using namespace std;
using namespace AscendC;

template <typename T, typename SrcOffsetT, int32_t MD>
class KernelScatter {
public:
    __aicore__ inline KernelScatter() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
        __gm__ uint8_t* srcOffsetGm, const uint32_t dstBaseAddr, const uint32_t count, const uint64_t maskVal,
         const uint64_t maskLow, const uint64_t maskHigh, const uint32_t srcRep)
    {

        src0_global.SetGlobalBuffer((__gm__ T*)srcGm);
        src1_global.SetGlobalBuffer((__gm__ SrcOffsetT*)srcOffsetGm);
        dst_global.SetGlobalBuffer((__gm__ T*)dstGm);


        pipe.InitBuffer(inQueueSrc0, 1, count * sizeof(T));
        pipe.InitBuffer(inQueueSrc1, 1, count * sizeof(SrcOffsetT));
        pipe.InitBuffer(outQueue, 1, count * sizeof(T));
        baseAddr = dstBaseAddr;
        counts = count;
        mask = maskVal;
        maskArr[0] = maskLow;
        maskArr[1] = maskHigh;
        srcRepStride = (uint8_t)srcRep;
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
        LocalTensor<T> src0Local = inQueueSrc0.AllocTensor<T>();
        LocalTensor<SrcOffsetT> src1Local = inQueueSrc1.AllocTensor<SrcOffsetT>();
        DataCopy(src0Local, src0_global, counts);
        DataCopy(src1Local, src1_global, counts);
        inQueueSrc0.EnQue(src0Local);
        inQueueSrc1.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> src0Local = inQueueSrc0.DeQue<T>();
        LocalTensor<SrcOffsetT> src1Local = inQueueSrc1.DeQue<SrcOffsetT>();
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        uint32_t oneBlkElems = 32/sizeof(T);
        uint32_t oneRepSize = oneBlkElems*srcRepStride;
        if (sizeof(T) == 1) {
            oneRepSize = 128;
        }
        uint32_t rep = (counts-1+oneRepSize)/oneRepSize;
        uint32_t baseAddr = 0;
        if constexpr (MD == 0) {
            Scatter(dstLocal, src0Local, src1Local, baseAddr, counts);
        } else if constexpr (MD == 1) {
            Scatter(dstLocal, src0Local, src1Local, baseAddr, mask, rep, srcRepStride);
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, 144);
            Scatter(dstLocal, src0Local, src1Local, baseAddr, AscendC::MASK_PLACEHOLDER, rep, srcRepStride);
            AscendC::ResetMask();
            Scatter(dstLocal, src0Local, src1Local, baseAddr, mask, rep, srcRepStride);
            AscendC::SetMaskNorm();
        } else if constexpr (MD == 2) {
            Scatter(dstLocal, src0Local, src1Local, baseAddr, maskArr, rep, srcRepStride);
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, 144);
            Scatter(dstLocal, src0Local, src1Local, baseAddr, AscendC::MASK_PLACEHOLDER_LIST, rep, srcRepStride);
            AscendC::ResetMask();
            Scatter(dstLocal, src0Local, src1Local, baseAddr, maskArr, rep, srcRepStride);
            AscendC::SetMaskNorm();
        }
        outQueue.EnQue(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
        inQueueSrc1.FreeTensor(src1Local);

    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        DataCopy(dst_global, dstLocal, counts);
        outQueue.FreeTensor(dstLocal);
    }
private:
    GlobalTensor<T> src0_global;
    GlobalTensor<SrcOffsetT> src1_global;
    GlobalTensor<T> dst_global;
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrc0;
    TQue<TPosition::VECIN, 1> inQueueSrc1;
    TQue<TPosition::VECOUT, 1> outQueue;

    uint64_t mask;
    uint64_t maskArr[2];
    uint32_t counts;
    uint8_t srcRepStride;
    uint32_t baseAddr = 0;
};
struct VecScatterParams {
    void (*CallFunc)();
};

template<typename T, int32_t MODE>
void RunVecScatter() {
    int byte_size = sizeof(T);
    int offset_byte_size = sizeof(uint32_t);
    uint32_t shape_size = 1024;
    uint64_t mask = 32;
    uint64_t maskL = 32;
    uint64_t maskH = 0;
    uint8_t dstGm[shape_size * byte_size] = {0};
    uint8_t src0Gm[shape_size * byte_size] = {0};
    uint8_t src1Gm[shape_size * offset_byte_size] = {0};
    uint32_t dstBaseAddr = 0;

    uint32_t srcRep;
    if constexpr (std::is_same<T, uint8_t>::value||std::is_same<T, int8_t>::value) {
        srcRep = 4;
    } else {
        srcRep = 8;
    }
    KernelScatter<T, uint32_t, MODE> op;
    op.Init(dstGm, src0Gm, src1Gm, dstBaseAddr, shape_size, mask, maskL, maskH, srcRep);
    op.Process();
}

class VecScatterTestsuite : public testing::Test, public testing::WithParamInterface<VecScatterParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(VecScatterTestCase, VecScatterTestsuite,
    ::testing::Values(VecScatterParams { RunVecScatter<uint8_t, 0> },
                        VecScatterParams { RunVecScatter<int8_t, 0> },
                        VecScatterParams { RunVecScatter<uint16_t, 0> },
                        VecScatterParams { RunVecScatter<int16_t, 0> },
                        VecScatterParams { RunVecScatter<bfloat16_t, 0> },
                        VecScatterParams { RunVecScatter<half, 0> },
                        VecScatterParams { RunVecScatter<uint32_t, 0> },
                        VecScatterParams { RunVecScatter<int32_t, 0> },
                        VecScatterParams { RunVecScatter<float, 0> },
                        VecScatterParams { RunVecScatter<uint64_t, 0> },
                        VecScatterParams { RunVecScatter<int64_t, 0> },
                        VecScatterParams { RunVecScatter<uint16_t, 1> },
                        VecScatterParams { RunVecScatter<int16_t, 1> },
                        VecScatterParams { RunVecScatter<bfloat16_t, 1> },
                        VecScatterParams { RunVecScatter<half, 1> },
                        VecScatterParams { RunVecScatter<uint32_t, 1> },
                        VecScatterParams { RunVecScatter<int32_t, 1> },
                        VecScatterParams { RunVecScatter<float, 1> },
                        VecScatterParams { RunVecScatter<uint64_t, 1> },
                        VecScatterParams { RunVecScatter<int64_t, 1> },
                        VecScatterParams { RunVecScatter<uint16_t, 2> },
                        VecScatterParams { RunVecScatter<int16_t, 2> },
                        VecScatterParams { RunVecScatter<bfloat16_t, 2> },
                        VecScatterParams { RunVecScatter<half, 2> },
                        VecScatterParams { RunVecScatter<uint32_t, 2> },
                        VecScatterParams { RunVecScatter<int32_t, 2> },
                        VecScatterParams { RunVecScatter<float, 2> },
                        VecScatterParams { RunVecScatter<uint64_t, 2> },
                        VecScatterParams { RunVecScatter<int64_t, 2> }
                      ));

TEST_P(VecScatterTestsuite, VecScatterTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}