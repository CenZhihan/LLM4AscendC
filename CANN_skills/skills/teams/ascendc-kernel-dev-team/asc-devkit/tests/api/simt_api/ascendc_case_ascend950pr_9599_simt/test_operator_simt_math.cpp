/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <cmath>
#include <gtest/gtest.h>
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;
using namespace AscendC::Simt;

#define THREAD_DIM 128

template <typename T>
class KernelSimtUintDivCompute {
    public:
        __aicore__ KernelSimtUintDivCompute() {}
        __aicore__ inline void Process(__gm__ T* dst, T n, T magic, T shift);
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelSimtUintDivComputeCompute(__gm__ T* dst, T n, T magic, T shift)
{
    for (int idx=GetThreadIdx<0>()+block_idx*GetThreadNum<0>(); idx < THREAD_DIM; idx+=block_num*GetThreadNum<0>()) {
        dst[idx] = UintDiv(n, magic, shift);
    }
}

template <typename T>
__aicore__ inline void KernelSimtUintDivCompute<T>::Process(__gm__ T* dst, T n, T magic, T shift)
{
    AscendC::Simt::VF_CALL<KernelSimtUintDivComputeCompute<T>>(AscendC::Simt::Dim3(THREAD_DIM, 1, 1), dst, n, magic, shift);
}

struct SimtUint64DivComputeParams {
    uint64_t n;
    uint64_t d;
};

class SimtUint64DivComputeTestsuite : public testing::Test, public testing::WithParamInterface<SimtUint64DivComputeParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(SimtUint64DivComputeTestCase, SimtUint64DivComputeTestsuite,
    ::testing::Values(SimtUint64DivComputeParams {.n = 10, .d = 5},
        SimtUint64DivComputeParams {.n = 5, .d = 10},
        SimtUint64DivComputeParams {.n = 5, .d = 5},
        SimtUint64DivComputeParams {.n = 0x7fffffffffffffff, .d = 0x7fffffffffffffff},
        SimtUint64DivComputeParams {.n = 0x7ffffffffffffffe, .d = 0x7fffffffffffffff},
        SimtUint64DivComputeParams {.n = 0x7fffffffffffffff, .d = 0x7ffffffffffffffe},
        SimtUint64DivComputeParams {.n = 1, .d = 0x7fffffffffffffff},
        SimtUint64DivComputeParams {.n = 0x7fffffffffffffff, .d = 1},
        SimtUint64DivComputeParams {.n = 0x0102030405060708, .d = 1}
    ));

TEST_P(SimtUint64DivComputeTestsuite, SimtUint64DivComputeTestCase)
{
    auto param = GetParam();

    int typeSize = 8;
    int shapeSize = THREAD_DIM;
    uint8_t dstGM[shapeSize * typeSize] = {0};

    KernelSimtUintDivCompute<uint64_t> op;
    uint64_t magic = 0;
    uint64_t shift = 0;
    GetUintDivMagicAndShift(magic, shift, param.d);
    printf("magic: %lu, shift: %lu\n", magic, shift);

    op.Process((__gm__ uint64_t*)dstGM, param.n, magic, shift);

    for (int i = 0; i < 256; i += 2) {
        if (i % 16 == 0 && i != 0) {
            printf("\n");
        }
        printf("%02x%02x ", dstGM[i], dstGM[i+1]);
    }

    uint64_t expectValues[shapeSize] = {0};
    for (int i = 0; i < shapeSize; i += 1) {
        expectValues[i] = param.n / param.d;
    }

    for (int i = 0; i < shapeSize; i += 1) {
        ASSERT_EQ(static_cast<uint8_t>(expectValues[i] & 0xFF), dstGM[i*typeSize]);
        ASSERT_EQ(static_cast<uint8_t>((expectValues[i] >> 8) & 0xFF), dstGM[i*typeSize+1]);
        ASSERT_EQ(static_cast<uint8_t>((expectValues[i] >> 16) & 0xFF), dstGM[i*typeSize+2]);
        ASSERT_EQ(static_cast<uint8_t>((expectValues[i] >> 24) & 0xFF), dstGM[i*typeSize+3]);
        ASSERT_EQ(static_cast<uint8_t>((expectValues[i] >> 32) & 0xFF), dstGM[i*typeSize+4]);
        ASSERT_EQ(static_cast<uint8_t>((expectValues[i] >> 40) & 0xFF), dstGM[i*typeSize+5]);
        ASSERT_EQ(static_cast<uint8_t>((expectValues[i] >> 48) & 0xFF), dstGM[i*typeSize+6]);
        ASSERT_EQ(static_cast<uint8_t>((expectValues[i] >> 56) & 0xFF), dstGM[i*typeSize+7]);
    }
}

struct SimtUint32DivComputeParams {
    uint32_t n;
    uint32_t d;
};

class SimtUint32DivComputeTestsuite : public testing::Test, public testing::WithParamInterface<SimtUint32DivComputeParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(SimtUintDiv32ComputeTestCase, SimtUint32DivComputeTestsuite,
    ::testing::Values(SimtUint32DivComputeParams {.n = 10, .d = 5},
        SimtUint32DivComputeParams {.n = 5, .d = 10},
        SimtUint32DivComputeParams {.n = 5, .d = 5},
        SimtUint32DivComputeParams {.n = 0x7fffffff, .d = 0x7fffffff},
        SimtUint32DivComputeParams {.n = 0x7ffffffe, .d = 0x7fffffff},
        SimtUint32DivComputeParams {.n = 0x7fffffff, .d = 0x7ffffffe},
        SimtUint32DivComputeParams {.n = 1, .d = 0x7fffffff},
        SimtUint32DivComputeParams {.n = 0x7fffffff, .d = 1},
        SimtUint32DivComputeParams {.n = 0x01020304, .d = 1}
    ));

TEST_P(SimtUint32DivComputeTestsuite, SimtUint32DivComputeTestCase)
{
    auto param = GetParam();

    int typeSize = 4;
    int shapeSize = THREAD_DIM;
    uint8_t dstGM[shapeSize * typeSize] = {0};

    KernelSimtUintDivCompute<uint32_t> op;
    uint32_t magic = 0;
    uint32_t shift = 0;
    GetUintDivMagicAndShift(magic, shift, param.d);
    printf("magic: %lu, shift: %lu\n", magic, shift);

    op.Process((__gm__ uint32_t*)dstGM, param.n, magic, shift);

    for (int i = 0; i < 256; i += 2) {
        if (i % 16 == 0 && i != 0) {
            printf("\n");
        }
        printf("%02x%02x ", dstGM[i], dstGM[i+1]);
    }

    uint32_t expectValues[shapeSize] = {0};
    for (int i = 0; i < shapeSize; i += 1) {
        expectValues[i] = param.n / param.d;
    }

    for (int i = 0; i < shapeSize; i += 1) {
        ASSERT_EQ(static_cast<uint8_t>(expectValues[i] & 0xFF), dstGM[i*typeSize]);
        ASSERT_EQ(static_cast<uint8_t>((expectValues[i] >> 8) & 0xFF), dstGM[i*typeSize+1]);
        ASSERT_EQ(static_cast<uint8_t>((expectValues[i] >> 16) & 0xFF), dstGM[i*typeSize+2]);
        ASSERT_EQ(static_cast<uint8_t>((expectValues[i] >> 24) & 0xFF), dstGM[i*typeSize+3]);
    }
}
