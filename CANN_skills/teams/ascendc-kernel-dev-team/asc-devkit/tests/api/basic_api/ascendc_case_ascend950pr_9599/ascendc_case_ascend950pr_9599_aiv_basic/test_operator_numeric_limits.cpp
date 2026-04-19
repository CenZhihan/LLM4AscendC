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

template <uint32_t category, typename T>
class KernelVecLimits {
public:
    __aicore__ inline KernelVecLimits()
    {}
    __aicore__ inline void Init(GM_ADDR dstGm, uint32_t dstCount)
    {
        count = dstCount;
        const int alginSize = AscendC::GetDataBlockSizeInBytes() / sizeof(T);
        dstSize = (dstCount + 256 + alginSize - 1) / alginSize * alginSize;
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), dstSize);
        pipe.InitBuffer(outQueue, 1, dstSize * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        AscendC::Duplicate(dstLocal, static_cast<T>(0), count + 256);
        // ["Max", "Lowest", "Min", "Infinity", "NegativeInfinity", "QuietNaN", "SignalingNaN", "DeNormMin"]
        if constexpr (category == 0) {
            AscendC::NumericLimits<T>::Max(dstLocal, count);
        } else if constexpr (category == 1) {
            AscendC::NumericLimits<T>::Lowest(dstLocal, count);
        } else if constexpr (category == 2) {
            AscendC::NumericLimits<T>::Min(dstLocal, count);
        } else if constexpr (category == 3) {
            AscendC::NumericLimits<T>::Infinity(dstLocal, count);
        } else if constexpr (category == 4) {
            AscendC::NumericLimits<T>::NegativeInfinity(dstLocal, count);
        } else if constexpr (category == 5) {
            AscendC::NumericLimits<T>::QuietNaN(dstLocal, count);
        } else if constexpr (category == 6) {
            AscendC::NumericLimits<T>::SignalingNaN(dstLocal, count);
        } else if constexpr (category == 7) {
            AscendC::NumericLimits<T>::DeNormMin(dstLocal, count);
        }
        outQueue.EnQue<T>(dstLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        AscendC::DataCopy(dstGlobal, dstLocal, dstSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::GlobalTensor<T> dstGlobal;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    uint32_t count;
    uint32_t dstSize;
};

template <uint32_t category, typename T>
__global__ __aicore__ void MainKernelNumericLimits(uint8_t *dstGm, uint32_t dstSize)
{
    KernelVecLimits<category, T> op;
    op.Init(dstGm, dstSize);
    op.Process();
}

struct NumericLimitsTestParams {
    void (*cal_func)(uint8_t *, uint32_t);
    uint32_t dstSize;
};

class NumericLimitsTestsuite : public testing::Test, public testing::WithParamInterface<NumericLimitsTestParams> {};

INSTANTIATE_TEST_CASE_P(NumericLimitsTestCase, NumericLimitsTestsuite,
    ::testing::Values(NumericLimitsTestParams{MainKernelNumericLimits<0, uint8_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<0, int8_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<0, uint16_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<0, int16_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<0, uint32_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<0, int32_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<0, half>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<0, bfloat16_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<0, float>, 257},

        NumericLimitsTestParams{MainKernelNumericLimits<1, uint8_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<1, int8_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<1, uint16_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<1, int16_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<1, uint32_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<1, int32_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<1, half>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<1, bfloat16_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<1, float>, 257},

        NumericLimitsTestParams{MainKernelNumericLimits<2, uint8_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<2, int8_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<2, uint16_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<2, int16_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<2, uint32_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<2, int32_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<2, half>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<2, bfloat16_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<2, float>, 257},

        NumericLimitsTestParams{MainKernelNumericLimits<3, half>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<3, bfloat16_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<3, float>, 257},

        NumericLimitsTestParams{MainKernelNumericLimits<4, half>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<4, bfloat16_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<4, float>, 257},

        NumericLimitsTestParams{MainKernelNumericLimits<5, half>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<5, bfloat16_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<5, float>, 257},

        NumericLimitsTestParams{MainKernelNumericLimits<6, half>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<6, bfloat16_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<6, float>, 257},

        NumericLimitsTestParams{MainKernelNumericLimits<7, half>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<7, bfloat16_t>, 257},
        NumericLimitsTestParams{MainKernelNumericLimits<7, float>, 257}));

TEST_P(NumericLimitsTestsuite, NumericLimitsTestCase)
{
    auto param = GetParam();
    uint8_t dstGm[param.dstSize] = {0};
    param.cal_func(dstGm, param.dstSize);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
