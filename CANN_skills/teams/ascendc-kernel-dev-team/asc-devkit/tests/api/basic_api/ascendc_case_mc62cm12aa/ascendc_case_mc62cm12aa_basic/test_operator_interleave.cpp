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

template <uint32_t Category, typename T>
class KernelVecInterleave {
public:
    __aicore__ inline KernelVecInterleave()
    {}
    __aicore__ inline void Init(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dst0Gm, GM_ADDR dst1Gm, uint32_t dstCount)
    {
        count = dstCount;
        const int alginSize = AscendC::GetDataBlockSizeInBytes() / sizeof(T);
        dstSize = (dstCount + 256 + alginSize - 1) / alginSize * alginSize;
        src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src0Gm), dstSize);
        src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src1Gm), dstSize);
        dst0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst0Gm), dstSize);
        dst1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst1Gm), dstSize);
        pipe.InitBuffer(in0Queue, 1, dstSize * sizeof(T));
        pipe.InitBuffer(in1Queue, 1, dstSize * sizeof(T));
        pipe.InitBuffer(out0Queue, 1, dstSize * sizeof(T));
        pipe.InitBuffer(out1Queue, 1, dstSize * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> src0Local = in0Queue.DeQue<T>();
        AscendC::LocalTensor<T> src1Local = in1Queue.DeQue<T>();
        AscendC::LocalTensor<T> dst0Local = out0Queue.AllocTensor<T>();
        AscendC::LocalTensor<T> dst1Local = out1Queue.AllocTensor<T>();
        AscendC::Duplicate(dst0Local, T(0), count + 256);
        AscendC::Duplicate(dst1Local, T(0), count + 256);
        // ["Interleave", "DeInterleave", "DeInterleave2"]
        if constexpr (Category == 0) {
            AscendC::Interleave(dst0Local, dst1Local, src0Local, src1Local, count);
        } else if constexpr (Category == 1) {
            AscendC::DeInterleave(dst0Local, dst1Local, src0Local, src1Local, count);
        } else if constexpr (Category == 2) {
            AscendC::DeInterleave(dst0Local, dst1Local, src0Local, count);
        }
        out0Queue.EnQue<T>(dst0Local);
        out1Queue.EnQue<T>(dst1Local);
    }

    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<T> src0Local = in0Queue.AllocTensor<T>();
        AscendC::LocalTensor<T> src1Local = in1Queue.AllocTensor<T>();
        AscendC::DataCopy(src0Local, src0Global, dstSize);
        AscendC::DataCopy(src1Local, src1Global, dstSize);
        in0Queue.EnQue(src0Local);
        in1Queue.EnQue(src1Local);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dst0Local = out0Queue.DeQue<T>();
        AscendC::LocalTensor<T> dst1Local = out1Queue.DeQue<T>();
        AscendC::DataCopy(dst0Global, dst0Local, dstSize);
        AscendC::DataCopy(dst1Global, dst1Local, dstSize);
        out0Queue.FreeTensor(dst0Local);
        out1Queue.FreeTensor(dst1Local);
    }

private:
    AscendC::GlobalTensor<T> src0Global;
    AscendC::GlobalTensor<T> src1Global;
    AscendC::GlobalTensor<T> dst0Global;
    AscendC::GlobalTensor<T> dst1Global;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> in0Queue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> in1Queue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> out0Queue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> out1Queue;
    uint32_t count;
    uint32_t dstSize;
};

template <uint32_t Category, typename T>
__global__ __aicore__ void MainKernelInterleave(
    uint8_t *src0Gm, uint8_t *src1Gm, uint8_t *dst0Gm, uint8_t *dst1Gm, uint32_t dstSize)
{
    KernelVecInterleave<Category, T> op;
    op.Init(src0Gm, src1Gm, dst0Gm, dst1Gm, dstSize);
    op.Process();
}

struct InterleaveTestParams {
    void (*cal_func)(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint32_t);
    uint32_t dstSize;
};

class InterleaveTestsuite : public testing::Test, public testing::WithParamInterface<InterleaveTestParams> {};

INSTANTIATE_TEST_CASE_P(InterleaveTestCase, InterleaveTestsuite,
    ::testing::Values(InterleaveTestParams{MainKernelInterleave<0, uint8_t>, 2},
        InterleaveTestParams{MainKernelInterleave<0, int8_t>, 2},
        InterleaveTestParams{MainKernelInterleave<0, uint16_t>, 2},
        InterleaveTestParams{MainKernelInterleave<0, int16_t>, 2},
        InterleaveTestParams{MainKernelInterleave<0, uint32_t>, 2},
        InterleaveTestParams{MainKernelInterleave<0, int32_t>, 2},
        InterleaveTestParams{MainKernelInterleave<0, uint64_t>, 2},
        InterleaveTestParams{MainKernelInterleave<0, int64_t>, 2},
        InterleaveTestParams{MainKernelInterleave<0, half>, 2},
        InterleaveTestParams{MainKernelInterleave<0, bfloat16_t>, 2},
        InterleaveTestParams{MainKernelInterleave<0, float>, 2},

        InterleaveTestParams{MainKernelInterleave<0, uint8_t>, 256},
        InterleaveTestParams{MainKernelInterleave<0, int8_t>, 256},
        InterleaveTestParams{MainKernelInterleave<0, uint16_t>, 256},
        InterleaveTestParams{MainKernelInterleave<0, int16_t>, 256},
        InterleaveTestParams{MainKernelInterleave<0, uint32_t>, 256},
        InterleaveTestParams{MainKernelInterleave<0, int32_t>, 256},
        InterleaveTestParams{MainKernelInterleave<0, uint64_t>, 256},
        InterleaveTestParams{MainKernelInterleave<0, int64_t>, 256},
        InterleaveTestParams{MainKernelInterleave<0, half>, 256},
        InterleaveTestParams{MainKernelInterleave<0, bfloat16_t>, 256},
        InterleaveTestParams{MainKernelInterleave<0, float>, 256},

        InterleaveTestParams{MainKernelInterleave<0, uint8_t>, 258},
        InterleaveTestParams{MainKernelInterleave<0, int8_t>, 258},
        InterleaveTestParams{MainKernelInterleave<0, uint16_t>, 258},
        InterleaveTestParams{MainKernelInterleave<0, int16_t>, 258},
        InterleaveTestParams{MainKernelInterleave<0, uint32_t>, 258},
        InterleaveTestParams{MainKernelInterleave<0, int32_t>, 258},
        InterleaveTestParams{MainKernelInterleave<0, uint64_t>, 258},
        InterleaveTestParams{MainKernelInterleave<0, int64_t>, 258},
        InterleaveTestParams{MainKernelInterleave<0, half>, 258},
        InterleaveTestParams{MainKernelInterleave<0, bfloat16_t>, 258},
        InterleaveTestParams{MainKernelInterleave<0, float>, 258},

        InterleaveTestParams{MainKernelInterleave<1, uint8_t>, 2},
        InterleaveTestParams{MainKernelInterleave<1, int8_t>, 2},
        InterleaveTestParams{MainKernelInterleave<1, uint16_t>, 2},
        InterleaveTestParams{MainKernelInterleave<1, int16_t>, 2},
        InterleaveTestParams{MainKernelInterleave<1, uint32_t>, 2},
        InterleaveTestParams{MainKernelInterleave<1, int32_t>, 2},
        InterleaveTestParams{MainKernelInterleave<1, uint64_t>, 2},
        InterleaveTestParams{MainKernelInterleave<1, int64_t>, 2},
        InterleaveTestParams{MainKernelInterleave<1, half>, 2},
        InterleaveTestParams{MainKernelInterleave<1, bfloat16_t>, 2},
        InterleaveTestParams{MainKernelInterleave<1, float>, 2},

        InterleaveTestParams{MainKernelInterleave<1, uint8_t>, 256},
        InterleaveTestParams{MainKernelInterleave<1, int8_t>, 256},
        InterleaveTestParams{MainKernelInterleave<1, uint16_t>, 256},
        InterleaveTestParams{MainKernelInterleave<1, int16_t>, 256},
        InterleaveTestParams{MainKernelInterleave<1, uint32_t>, 256},
        InterleaveTestParams{MainKernelInterleave<1, int32_t>, 256},
        InterleaveTestParams{MainKernelInterleave<1, uint64_t>, 256},
        InterleaveTestParams{MainKernelInterleave<1, int64_t>, 256},
        InterleaveTestParams{MainKernelInterleave<1, half>, 256},
        InterleaveTestParams{MainKernelInterleave<1, bfloat16_t>, 256},
        InterleaveTestParams{MainKernelInterleave<1, float>, 256},

        InterleaveTestParams{MainKernelInterleave<1, uint8_t>, 258},
        InterleaveTestParams{MainKernelInterleave<1, int8_t>, 258},
        InterleaveTestParams{MainKernelInterleave<1, uint16_t>, 258},
        InterleaveTestParams{MainKernelInterleave<1, int16_t>, 258},
        InterleaveTestParams{MainKernelInterleave<1, uint32_t>, 258},
        InterleaveTestParams{MainKernelInterleave<1, int32_t>, 258},
        InterleaveTestParams{MainKernelInterleave<1, uint64_t>, 258},
        InterleaveTestParams{MainKernelInterleave<1, int64_t>, 258},
        InterleaveTestParams{MainKernelInterleave<1, half>, 258},
        InterleaveTestParams{MainKernelInterleave<1, bfloat16_t>, 258},
        InterleaveTestParams{MainKernelInterleave<1, float>, 258},

        InterleaveTestParams{MainKernelInterleave<2, uint8_t>, 2},
        InterleaveTestParams{MainKernelInterleave<2, int8_t>, 2},
        InterleaveTestParams{MainKernelInterleave<2, uint16_t>, 2},
        InterleaveTestParams{MainKernelInterleave<2, int16_t>, 2},
        InterleaveTestParams{MainKernelInterleave<2, uint32_t>, 2},
        InterleaveTestParams{MainKernelInterleave<2, int32_t>, 2},
        InterleaveTestParams{MainKernelInterleave<2, uint64_t>, 2},
        InterleaveTestParams{MainKernelInterleave<2, int64_t>, 2},
        InterleaveTestParams{MainKernelInterleave<2, half>, 2},
        InterleaveTestParams{MainKernelInterleave<2, bfloat16_t>, 2},
        InterleaveTestParams{MainKernelInterleave<2, float>, 2},

        InterleaveTestParams{MainKernelInterleave<2, uint8_t>, 256},
        InterleaveTestParams{MainKernelInterleave<2, int8_t>, 256},
        InterleaveTestParams{MainKernelInterleave<2, uint16_t>, 256},
        InterleaveTestParams{MainKernelInterleave<2, int16_t>, 256},
        InterleaveTestParams{MainKernelInterleave<2, uint32_t>, 256},
        InterleaveTestParams{MainKernelInterleave<2, int32_t>, 256},
        InterleaveTestParams{MainKernelInterleave<2, uint64_t>, 256},
        InterleaveTestParams{MainKernelInterleave<2, int64_t>, 256},
        InterleaveTestParams{MainKernelInterleave<2, half>, 256},
        InterleaveTestParams{MainKernelInterleave<2, bfloat16_t>, 256},
        InterleaveTestParams{MainKernelInterleave<2, float>, 256},

        InterleaveTestParams{MainKernelInterleave<2, uint8_t>, 258},
        InterleaveTestParams{MainKernelInterleave<2, int8_t>, 258},
        InterleaveTestParams{MainKernelInterleave<2, uint16_t>, 258},
        InterleaveTestParams{MainKernelInterleave<2, int16_t>, 258},
        InterleaveTestParams{MainKernelInterleave<2, uint32_t>, 258},
        InterleaveTestParams{MainKernelInterleave<2, int32_t>, 258},
        InterleaveTestParams{MainKernelInterleave<2, uint64_t>, 258},
        InterleaveTestParams{MainKernelInterleave<2, int64_t>, 258},
        InterleaveTestParams{MainKernelInterleave<2, half>, 258},
        InterleaveTestParams{MainKernelInterleave<2, bfloat16_t>, 258},
        InterleaveTestParams{MainKernelInterleave<2, float>, 258}));

TEST_P(InterleaveTestsuite, InterleaveTestCase)
{
    auto param = GetParam();
    const int alginSize = 32;
    auto dstSize = (param.dstSize + 256 + alginSize - 1) / alginSize * alginSize;
    uint8_t* src0Gm = new uint8_t[dstSize * 8] {0};
    uint8_t* src1Gm = new uint8_t[dstSize * 8] {0};
    uint8_t* dst0Gm = new uint8_t[dstSize * 8] {0};
    uint8_t* dst1Gm = new uint8_t[dstSize * 8] {0};
    param.cal_func(src0Gm, src1Gm, dst0Gm, dst1Gm, param.dstSize);
    for (int32_t i = 0; i < (sizeof(dst0Gm) / sizeof(dst0Gm[0])); i++) {
        EXPECT_EQ(dst0Gm[i], 0x00);
        EXPECT_EQ(dst1Gm[i], 0x00);
    }
    delete [] src0Gm;
    delete [] src1Gm;
    delete [] dst0Gm;
    delete [] dst1Gm;
}
