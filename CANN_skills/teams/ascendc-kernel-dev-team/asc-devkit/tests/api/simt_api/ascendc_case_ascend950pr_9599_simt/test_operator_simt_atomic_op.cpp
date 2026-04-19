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

#define THREAD_DIM 128

template <typename T>
class KernelAtomic {
public:
    __aicore__ KernelAtomic() {}

public:
    __aicore__ inline void VfCallProcess(__gm__ T* dst, T value, T compare, const int num, const int mode);
    __aicore__ inline void Process(__gm__ T* dst, T value, T compare, const int num, const int mode);
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelAtomicCompute(__gm__ T* dst, T value, T compare, const int num, const int mode)
{
    for (int idx = AscendC::Simt::GetThreadIdx<0>() + block_idx * AscendC::Simt::GetThreadNum<0>(); idx < num; idx += block_num * AscendC::Simt::GetThreadNum<0>()) {
        if (mode == 0) {
            AscendC::Simt::AtomicAdd(dst, value);
        } else if (mode == 1) {
            AscendC::Simt::AtomicSub(dst, value);
        } else if (mode == 2) {
            AscendC::Simt::AtomicExch(dst, value);
        } else if (mode == 3) {
            AscendC::Simt::AtomicMax(dst, value);
        } else if (mode == 4) {
            AscendC::Simt::AtomicMin(dst, value);
        } else if (mode == 5) {
            AscendC::Simt::AtomicInc(dst, value);
        } else if (mode == 6) {
            AscendC::Simt::AtomicDec(dst, value);
        } else if (mode == 7) {
            AscendC::Simt::AtomicCas(dst, compare, value);
        } else if (mode == 8) {
            AscendC::Simt::AtomicAnd(dst, value);
        } else if (mode == 9) {
            AscendC::Simt::AtomicOr(dst, value);
        } else if (mode == 10) {
            AscendC::Simt::AtomicXor(dst, value);
        }
    }
}

template <typename T>
__aicore__ inline void KernelAtomic<T>::Process(__gm__ T* dst, T value, T compare, const int num, const int mode)
{
    AscendC::Simt::VF_CALL<KernelAtomicCompute<T>>(AscendC::Simt::Dim3(THREAD_DIM, 1, 1), dst, value, compare, num, mode);
}

template <typename T>
__simt_vf__ inline void VfCallProcessStub(__gm__ T *dst, T value, T compare, int num, int mode)
{
    for (int idx = AscendC::Simt::GetThreadIdx<0>() + block_idx * AscendC::Simt::GetThreadNum<0>(); idx < num; idx += block_num * AscendC::Simt::GetThreadNum<0>()) {
        if (mode == 0) {
            AscendC::Simt::AtomicAdd(dst, value);
        } else if (mode == 1) {
            AscendC::Simt::AtomicSub(dst, value);
        } else if (mode == 2) {
            AscendC::Simt::AtomicExch(dst, value);
        } else if (mode == 3) {
            AscendC::Simt::AtomicMax(dst, value);
        } else if (mode == 4) {
            AscendC::Simt::AtomicMin(dst, value);
        } else if (mode == 5) {
            AscendC::Simt::AtomicInc(dst, value);
        } else if (mode == 6) {
            AscendC::Simt::AtomicDec(dst, value);
        } else if (mode == 7) {
            AscendC::Simt::AtomicCas(dst, compare, value);
        } else if (mode == 8) {
            AscendC::Simt::AtomicAnd(dst, value);
        } else if (mode == 9) {
            AscendC::Simt::AtomicOr(dst, value);
        } else if (mode == 10) {
            AscendC::Simt::AtomicXor(dst, value);
        }
    }
}

template <typename T>
__aicore__ inline void KernelAtomic<T>::VfCallProcess(__gm__ T* dst, T value, T compare, const int num, const int mode)
{
    AscendC::Simt::VF_CALL<VfCallProcessStub<T>>(cce::dim3(THREAD_DIM), dst, value, compare, num, mode);
}

struct AtomicOpParams {
    int value = 100;
    int compare = 1000;
    int num = 128;
    int mode;
};

class AtomicOpTestsuite : public testing::Test, public testing::WithParamInterface<AtomicOpParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(AtomicOpTestCase, AtomicOpTestsuite,
                        ::testing::Values(AtomicOpParams{.mode = 0}, AtomicOpParams{.mode = 1},
                                          AtomicOpParams{.mode = 2}, AtomicOpParams{.mode = 3},
                                          AtomicOpParams{.mode = 4}, AtomicOpParams{.mode = 5},
                                          AtomicOpParams{.mode = 6}, AtomicOpParams{.mode = 7},
                                          AtomicOpParams{.mode = 8}, AtomicOpParams{.mode = 9},
                                          AtomicOpParams{.mode = 10}));

TEST_P(AtomicOpTestsuite, AtomicOpTestCase)
{
    auto param = GetParam();
    int value = param.value;
    int compare = param.compare;
    int num = param.num;
    int mode = param.mode;

    int fpByteSize = 4;

    uint8_t dstGm[fpByteSize] = {0};
    KernelAtomic<uint32_t> op;
    op.Process((__gm__ uint32_t*)dstGm, value, compare, num, mode);

    uint32_t expectValue = 0;
    for (int i = 0; i < num; i += 1) {
        if (mode == 0) {
            expectValue = expectValue + value;
        } else if (mode == 1) {
            expectValue = expectValue - value;
        } else if (mode == 2) {
            expectValue = value;
        } else if (mode == 3) {
            expectValue = expectValue > value ? expectValue : value;
        } else if (mode == 4) {
            expectValue = expectValue < value ? expectValue : value;
        } else if (mode == 5) {
            expectValue = expectValue >= value ? 0 : expectValue + 1;
        } else if (mode == 6) {
            expectValue = expectValue == 0 || expectValue > value ? value : expectValue - 1;
        } else if (mode == 7) {
            expectValue = expectValue == compare ? value : expectValue;
        } else if (mode == 8) {
            expectValue = expectValue & value;
        } else if (mode == 9) {
            expectValue = expectValue | value;
        } else if (mode == 10) {
            expectValue = expectValue ^ value;
        }
    }

    printf("expect value: %lu\n", expectValue);

    ASSERT_EQ(static_cast<uint8_t>(expectValue & 0xFF), dstGm[0]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 8) & 0xFF), dstGm[1]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 16) & 0xFF), dstGm[2]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 24) & 0xFF), dstGm[3]);
}

TEST_P(AtomicOpTestsuite, AtomicOpVfCallTestCase)
{
    auto param = GetParam();
    int32_t value = param.value;
    int32_t compare = param.compare;
    int32_t num = param.num;
    int32_t mode = param.mode;

    int32_t fpByteSize = 4;

    uint8_t dstGm[fpByteSize] = {0};
    KernelAtomic<uint32_t> op;
    op.VfCallProcess((__gm__ uint32_t*)dstGm, value, compare, num, mode);

    uint32_t expectValue = 0;
    for (int i = 0; i < num; i += 1) {
        if (mode == 0) {
            expectValue = expectValue + value;
        } else if (mode == 1) {
            expectValue = expectValue - value;
        } else if (mode == 2) {
            expectValue = value;
        } else if (mode == 3) {
            expectValue = expectValue > value ? expectValue : value;
        } else if (mode == 4) {
            expectValue = expectValue < value ? expectValue : value;
        } else if (mode == 5) {
            expectValue = expectValue >= value ? 0 : expectValue + 1;
        } else if (mode == 6) {
            expectValue = expectValue == 0 || expectValue > value ? value : expectValue - 1;
        } else if (mode == 7) {
            expectValue = expectValue == compare ? value : expectValue;
        } else if (mode == 8) {
            expectValue = expectValue & value;
        } else if (mode == 9) {
            expectValue = expectValue | value;
        } else if (mode == 10) {
            expectValue = expectValue ^ value;
        }
    }

    printf("expect value: %lu\n", expectValue);

    ASSERT_EQ(static_cast<uint8_t>(expectValue & 0xFF), dstGm[0]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 8) & 0xFF), dstGm[1]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 16) & 0xFF), dstGm[2]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 24) & 0xFF), dstGm[3]);
}
