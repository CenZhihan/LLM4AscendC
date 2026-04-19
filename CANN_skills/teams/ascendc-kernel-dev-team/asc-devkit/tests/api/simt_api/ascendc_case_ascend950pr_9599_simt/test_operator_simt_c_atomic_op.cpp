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
#include "simt_compiler_stub.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace std;
using namespace AscendC;

constexpr int THREAD_DIM = 128;

template <typename T>
class KernelAtomic {
public:
    __aicore__ KernelAtomic() {}

public:
    // uint32_t uint64_t
    __aicore__ inline void VfCallProcess(__gm__ T* dst, T value, T compare, const int num, const int mode);
    __aicore__ inline void Process(__gm__ T* dst, T value, T compare, const int num, const int mode);
    // int32_t int64_t
    __aicore__ inline void VfCallProcess2(__gm__ T* dst, T value, T compare, const int num, const int mode);
    __aicore__ inline void Process2(__gm__ T* dst, T value, T compare, const int num, const int mode);
    // float half2 bfloat16x2_t
    __aicore__ inline void VfCallProcess3(__gm__ T* dst, T value, T compare, const int num, const int mode);
    __aicore__ inline void Process3(__gm__ T* dst, T value, T compare, const int num, const int mode);
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelAtomicCompute(__gm__ T* dst, T value, T compare, const int num, const int mode)
{
    for (int idx = AscendC::Simt::GetThreadIdx<0>() + block_idx * AscendC::Simt::GetThreadNum<0>(); idx < num; idx += block_num * AscendC::Simt::GetThreadNum<0>()) {
        if (mode == 0) {
            asc_atomic_add(dst, value);
        } else if (mode == 1) {
            asc_atomic_sub(dst, value);
        } else if (mode == 2) {
            asc_atomic_exch(dst, value);
        } else if (mode == 3) {
            asc_atomic_max(dst, value);
        } else if (mode == 4) {
            asc_atomic_min(dst, value);
        } else if (mode == 5) {
            asc_atomic_inc(dst, value);
        } else if (mode == 6) {
            asc_atomic_dec(dst, value);
        } else if (mode == 7) {
            asc_atomic_cas(dst, compare, value);
        } else if (mode == 8) {
            asc_atomic_and(dst, value);
        } else if (mode == 9) {
            asc_atomic_or(dst, value);
        } else if (mode == 10) {
            asc_atomic_xor(dst, value);
        }
    }
}

template <typename T>
__aicore__ inline void KernelAtomic<T>::Process(__gm__ T* dst, T value, T compare, const int num, const int mode)
{
    asc_vf_call<KernelAtomicCompute<T>>(dim3(THREAD_DIM, 1, 1), dst, value, compare, num, mode);
}

template <typename T>
__simt_vf__ inline void VfCallProcessStub(__gm__ T *dst, T value, T compare, int num, int mode)
{
    for (int idx = AscendC::Simt::GetThreadIdx<0>() + block_idx * AscendC::Simt::GetThreadNum<0>(); idx < num; idx += block_num * AscendC::Simt::GetThreadNum<0>()) {
        if (mode == 0) {
            asc_atomic_add(dst, value);
        } else if (mode == 1) {
            asc_atomic_sub(dst, value);
        } else if (mode == 2) {
            asc_atomic_exch(dst, value);
        } else if (mode == 3) {
            asc_atomic_max(dst, value);
        } else if (mode == 4) {
            asc_atomic_min(dst, value);
        } else if (mode == 5) {
            asc_atomic_inc(dst, value);
        } else if (mode == 6) {
            asc_atomic_dec(dst, value);
        } else if (mode == 7) {
            asc_atomic_cas(dst, compare, value);
        } else if (mode == 8) {
            asc_atomic_and(dst, value);
        } else if (mode == 9) {
            asc_atomic_or(dst, value);
        } else if (mode == 10) {
            asc_atomic_xor(dst, value);
        }
    }
}

template <typename T>
__aicore__ inline void KernelAtomic<T>::VfCallProcess(__gm__ T* dst, T value, T compare, const int num, const int mode)
{
    asc_vf_call<VfCallProcessStub<T>>(cce::dim3(THREAD_DIM), dst, value, compare, num, mode);
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelAtomicCompute2(__gm__ T* dst, T value, T compare, const int num, const int mode)
{
    for (int idx = AscendC::Simt::GetThreadIdx<0>() + block_idx * AscendC::Simt::GetThreadNum<0>(); idx < num; idx += block_num * AscendC::Simt::GetThreadNum<0>()) {
        if (mode == 0) {
            asc_atomic_add(dst, value);
        } else if (mode == 1) {
            asc_atomic_sub(dst, value);
        } else if (mode == 2) {
            asc_atomic_exch(dst, value);
        } else if (mode == 3) {
            asc_atomic_max(dst, value);
        } else if (mode == 4) {
            asc_atomic_min(dst, value);
        } else if (mode == 7) {
            asc_atomic_cas(dst, compare, value);
        } else if (mode == 8) {
            asc_atomic_and(dst, value);
        } else if (mode == 9) {
            asc_atomic_or(dst, value);
        } else if (mode == 10) {
            asc_atomic_xor(dst, value);
        }
    }
}

template <typename T>
__aicore__ inline void KernelAtomic<T>::Process2(__gm__ T* dst, T value, T compare, const int num, const int mode)
{
    asc_vf_call<KernelAtomicCompute2<T>>(dim3(THREAD_DIM, 1, 1), dst, value, compare, num, mode);
}

template <typename T>
__simt_vf__ inline void VfCallProcessStub2(__gm__ T *dst, T value, T compare, int num, int mode)
{
    for (int idx = AscendC::Simt::GetThreadIdx<0>() + block_idx * AscendC::Simt::GetThreadNum<0>(); idx < num; idx += block_num * AscendC::Simt::GetThreadNum<0>()) {
        if (mode == 0) {
            asc_atomic_add(dst, value);
        } else if (mode == 1) {
            asc_atomic_sub(dst, value);
        } else if (mode == 2) {
            asc_atomic_exch(dst, value);
        } else if (mode == 3) {
            asc_atomic_max(dst, value);
        } else if (mode == 4) {
            asc_atomic_min(dst, value);
        } else if (mode == 7) {
            asc_atomic_cas(dst, compare, value);
        } else if (mode == 8) {
            asc_atomic_and(dst, value);
        } else if (mode == 9) {
            asc_atomic_or(dst, value);
        } else if (mode == 10) {
            asc_atomic_xor(dst, value);
        }
    }
}

template <typename T>
__aicore__ inline void KernelAtomic<T>::VfCallProcess2(__gm__ T* dst, T value, T compare, const int num, const int mode)
{
    asc_vf_call<VfCallProcessStub2<T>>(cce::dim3(THREAD_DIM), dst, value, compare, num, mode);
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelAtomicCompute3(__gm__ T* dst, T value, T compare, const int num, const int mode)
{
    for (int idx = AscendC::Simt::GetThreadIdx<0>() + block_idx * AscendC::Simt::GetThreadNum<0>(); idx < num; idx += block_num * AscendC::Simt::GetThreadNum<0>()) {
        if (mode == 0) {
            asc_atomic_add(dst, value);
        } else if (mode == 1) {
            asc_atomic_sub(dst, value);
        } else if (mode == 3) {
            asc_atomic_max(dst, value);
        } else if (mode == 4) {
            asc_atomic_min(dst, value);
        } else if (mode == 2) {
            asc_atomic_exch(dst, value);
        } else if (mode == 7) {
            asc_atomic_cas(dst, compare, value);
        }
    }
}

template <typename T>
__aicore__ inline void KernelAtomic<T>::Process3(__gm__ T* dst, T value, T compare, const int num, const int mode)
{
    asc_vf_call<KernelAtomicCompute3<T>>(dim3(THREAD_DIM, 1, 1), dst, value, compare, num, mode);
}

template <typename T>
__simt_vf__ inline void VfCallProcessStub3(__gm__ T *dst, T value, T compare, int num, int mode)
{
    for (int idx = AscendC::Simt::GetThreadIdx<0>() + block_idx * AscendC::Simt::GetThreadNum<0>(); idx < num; idx += block_num * AscendC::Simt::GetThreadNum<0>()) {
        if (mode == 0) {
            asc_atomic_add(dst, value);
        } else if (mode == 1) {
            asc_atomic_sub(dst, value);
        } else if (mode == 3) {
            asc_atomic_max(dst, value);
        } else if (mode == 4) {
            asc_atomic_min(dst, value);
        } else if (mode == 2) {
            asc_atomic_exch(dst, value);
        } else if (mode == 7) {
            asc_atomic_cas(dst, compare, value);
        }
    }
}

template <typename T>
__aicore__ inline void KernelAtomic<T>::VfCallProcess3(__gm__ T* dst, T value, T compare, const int num, const int mode)
{
    asc_vf_call<VfCallProcessStub3<T>>(cce::dim3(THREAD_DIM), dst, value, compare, num, mode);
}

// ================================ Test int32_t start ================================
struct AtomicOpParams_int32 {
    int32_t value = 100;
    int32_t compare = 1000;
    int num = 128;
    int mode;
};

class AtomicOpTestsuite_int32 : public testing::Test, public testing::WithParamInterface<AtomicOpParams_int32> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(AtomicOpTestCase_int32, AtomicOpTestsuite_int32,
                        ::testing::Values(AtomicOpParams_int32{.mode = 0}, AtomicOpParams_int32{.mode = 1},
                                          AtomicOpParams_int32{.mode = 2}, AtomicOpParams_int32{.mode = 3},
                                          AtomicOpParams_int32{.mode = 4}, AtomicOpParams_int32{.mode = 7},
                                          AtomicOpParams_int32{.mode = 8}, AtomicOpParams_int32{.mode = 9},
                                          AtomicOpParams_int32{.mode = 10}));

TEST_P(AtomicOpTestsuite_int32, AtomicOpTestCase_int32)
{
    auto param = GetParam();
    int32_t value = param.value;
    int32_t compare = param.compare;
    int num = param.num;
    int mode = param.mode;

    int fpByteSize = 4;

    uint8_t dstGm[fpByteSize] = {0};
    KernelAtomic<int32_t> op;
    op.Process2((__gm__ int32_t*)dstGm, value, compare, num, mode);

    int32_t expectValue = 0;
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

    ASSERT_EQ(static_cast<uint8_t>(expectValue & 0xFF), dstGm[0]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 8) & 0xFF), dstGm[1]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 16) & 0xFF), dstGm[2]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 24) & 0xFF), dstGm[3]);
}

TEST_P(AtomicOpTestsuite_int32, AtomicOpVfCallTestCase_int32)
{
    auto param = GetParam();
    int32_t value = param.value;
    int32_t compare = param.compare;
    int num = param.num;
    int mode = param.mode;

    int32_t fpByteSize = 4;

    uint8_t dstGm[fpByteSize] = {0};
    KernelAtomic<int32_t> op;
    op.VfCallProcess2((__gm__ int32_t*)dstGm, value, compare, num, mode);

    int32_t expectValue = 0;
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

    ASSERT_EQ(static_cast<uint8_t>(expectValue & 0xFF), dstGm[0]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 8) & 0xFF), dstGm[1]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 16) & 0xFF), dstGm[2]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 24) & 0xFF), dstGm[3]);
}
// ================================ Test int32_t end ==================================

// ================================ Test uint32_t start ================================
struct AtomicOpParams_uint32 {
    uint32_t value = 100;
    uint32_t compare = 1000;
    int num = 128;
    int mode;
};

class AtomicOpTestsuite_uint32 : public testing::Test, public testing::WithParamInterface<AtomicOpParams_uint32> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(AtomicOpTestCase_uint32, AtomicOpTestsuite_uint32,
                        ::testing::Values(AtomicOpParams_uint32{.mode = 0}, AtomicOpParams_uint32{.mode = 1},
                                          AtomicOpParams_uint32{.mode = 2}, AtomicOpParams_uint32{.mode = 3},
                                          AtomicOpParams_uint32{.mode = 4}, AtomicOpParams_uint32{.mode = 5},
                                          AtomicOpParams_uint32{.mode = 6}, AtomicOpParams_uint32{.mode = 7},
                                          AtomicOpParams_uint32{.mode = 8}, AtomicOpParams_uint32{.mode = 9},
                                          AtomicOpParams_uint32{.mode = 10}));

TEST_P(AtomicOpTestsuite_uint32, AtomicOpTestCase_uint32)
{
    auto param = GetParam();
    uint32_t value = param.value;
    uint32_t compare = param.compare;
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

    ASSERT_EQ(static_cast<uint8_t>(expectValue & 0xFF), dstGm[0]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 8) & 0xFF), dstGm[1]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 16) & 0xFF), dstGm[2]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 24) & 0xFF), dstGm[3]);
}

TEST_P(AtomicOpTestsuite_uint32, AtomicOpVfCallTestCase_uint32)
{
    auto param = GetParam();
    uint32_t value = param.value;
    uint32_t compare = param.compare;
    int num = param.num;
    int mode = param.mode;

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

    ASSERT_EQ(static_cast<uint8_t>(expectValue & 0xFF), dstGm[0]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 8) & 0xFF), dstGm[1]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 16) & 0xFF), dstGm[2]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 24) & 0xFF), dstGm[3]);
}
// ================================ Test uint32_t end ==================================

// ================================ Test int64_t start ================================
struct AtomicOpParams_int64 {
    int64_t value = 100;
    int64_t compare = 1000;
    int num = 128;
    int mode;
};

class AtomicOpTestsuite_int64 : public testing::Test, public testing::WithParamInterface<AtomicOpParams_int64> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(AtomicOpTestCase_int64, AtomicOpTestsuite_int64,
                        ::testing::Values(AtomicOpParams_int64{.mode = 0}, AtomicOpParams_int64{.mode = 1},
                                          AtomicOpParams_int64{.mode = 2}, AtomicOpParams_int64{.mode = 3},
                                          AtomicOpParams_int64{.mode = 4}, AtomicOpParams_int64{.mode = 7},
                                          AtomicOpParams_int64{.mode = 8}, AtomicOpParams_int64{.mode = 9},
                                          AtomicOpParams_int64{.mode = 10}));

TEST_P(AtomicOpTestsuite_int64, AtomicOpTestCase_int64)
{
    auto param = GetParam();
    int64_t value = param.value;
    int64_t compare = param.compare;
    int num = param.num;
    int mode = param.mode;

    int fpByteSize = 8;

    uint8_t dstGm[fpByteSize] = {0};
    KernelAtomic<int64_t> op;
    op.Process2((__gm__ int64_t*)dstGm, value, compare, num, mode);

    int64_t expectValue = 0;
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

    ASSERT_EQ(static_cast<uint8_t>(expectValue & 0xFF), dstGm[0]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 8) & 0xFF), dstGm[1]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 16) & 0xFF), dstGm[2]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 24) & 0xFF), dstGm[3]);
}

TEST_P(AtomicOpTestsuite_int64, AtomicOpVfCallTestCase_int64)
{
    auto param = GetParam();
    int64_t value = param.value;
    int64_t compare = param.compare;
    int num = param.num;
    int mode = param.mode;

    int32_t fpByteSize = 8;

    uint8_t dstGm[fpByteSize] = {0};
    KernelAtomic<int64_t> op;
    op.VfCallProcess2((__gm__ int64_t*)dstGm, value, compare, num, mode);

    int64_t expectValue = 0;
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

    ASSERT_EQ(static_cast<uint8_t>(expectValue & 0xFF), dstGm[0]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 8) & 0xFF), dstGm[1]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 16) & 0xFF), dstGm[2]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 24) & 0xFF), dstGm[3]);
}
// ================================ Test int64_t end ==================================

// ================================ Test uint64_t start ================================
struct AtomicOpParams_uint64 {
    uint64_t value = 100;
    uint64_t compare = 1000;
    int num = 128;
    int mode;
};

class AtomicOpTestsuite_uint64 : public testing::Test, public testing::WithParamInterface<AtomicOpParams_uint64> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(AtomicOpTestCase_uint64, AtomicOpTestsuite_uint64,
                        ::testing::Values(AtomicOpParams_uint64{.mode = 0}, AtomicOpParams_uint64{.mode = 1},
                                          AtomicOpParams_uint64{.mode = 2}, AtomicOpParams_uint64{.mode = 3},
                                          AtomicOpParams_uint64{.mode = 4}, AtomicOpParams_uint64{.mode = 5},
                                          AtomicOpParams_uint64{.mode = 6}, AtomicOpParams_uint64{.mode = 7},
                                          AtomicOpParams_uint64{.mode = 8}, AtomicOpParams_uint64{.mode = 9},
                                          AtomicOpParams_uint64{.mode = 10}));

TEST_P(AtomicOpTestsuite_uint64, AtomicOpTestCase_uint64)
{
    auto param = GetParam();
    uint64_t value = param.value;
    uint64_t compare = param.compare;
    int num = param.num;
    int mode = param.mode;

    int fpByteSize = 8;

    uint8_t dstGm[fpByteSize] = {0};
    KernelAtomic<uint64_t> op;
    op.Process((__gm__ uint64_t*)dstGm, value, compare, num, mode);

    uint64_t expectValue = 0;
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

    ASSERT_EQ(static_cast<uint8_t>(expectValue & 0xFF), dstGm[0]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 8) & 0xFF), dstGm[1]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 16) & 0xFF), dstGm[2]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 24) & 0xFF), dstGm[3]);
}

TEST_P(AtomicOpTestsuite_uint64, AtomicOpVfCallTestCase_uint64)
{
    auto param = GetParam();
    uint64_t value = param.value;
    uint64_t compare = param.compare;
    int num = param.num;
    int mode = param.mode;

    int32_t fpByteSize = 8;

    uint8_t dstGm[fpByteSize] = {0};
    KernelAtomic<uint64_t> op;
    op.VfCallProcess((__gm__ uint64_t*)dstGm, value, compare, num, mode);

    uint64_t expectValue = 0;
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

    ASSERT_EQ(static_cast<uint8_t>(expectValue & 0xFF), dstGm[0]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 8) & 0xFF), dstGm[1]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 16) & 0xFF), dstGm[2]);
    ASSERT_EQ(static_cast<uint8_t>((expectValue >> 24) & 0xFF), dstGm[3]);
}
// ================================ Test uint64_t end ==================================

// ================================ Test float start ================================
struct AtomicOpParams_float {
    float value = 100.0f;
    float compare = 1000.0f;
    int32_t num = 128;
    int32_t mode;
};

class AtomicOpTestsuite_float : public testing::Test, public testing::WithParamInterface<AtomicOpParams_float> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(AtomicOpTestCase_float, AtomicOpTestsuite_float,
                        ::testing::Values(AtomicOpParams_float{.mode = 0}, AtomicOpParams_float{.mode = 1},
                                          AtomicOpParams_float{.mode = 2}, AtomicOpParams_float{.mode = 7},
                                          AtomicOpParams_float{.mode = 3}, AtomicOpParams_float{.mode = 4}));

TEST_P(AtomicOpTestsuite_float, AtomicOpTestCase_float)
{
    auto param = GetParam();
    float value = param.value;
    float compare = param.compare;
    int num = param.num;
    int mode = param.mode;

    int fpByteSize = 4;

    uint8_t dstGm[fpByteSize] = {0};
    KernelAtomic<float> op;
    op.Process3((__gm__ float*)dstGm, value, compare, num, mode);

    float expectValue = 0.0f;
    for (int i = 0; i < num; i += 1) {
        if (mode == 0) {
            expectValue = expectValue + value;
        } else if (mode == 1) {
            expectValue = expectValue - value;
        } else if (mode == 3) {
            expectValue = expectValue > value ? expectValue : value;
        } else if (mode == 4) {
            expectValue = expectValue < value ? expectValue : value;
        } else if (mode == 2) {
            expectValue = value;
        } else if (mode == 7) {
            expectValue = expectValue == compare ? value : expectValue;
        }
    }

    // uint32_t expectBits = *reinterpret_cast<uint32_t*>(&expectValue);
    uint32_t expectBits;
    memcpy(&expectBits, &expectValue, sizeof(expectBits));
    ASSERT_EQ(static_cast<uint8_t>(expectBits & 0xFF), dstGm[0]);
    ASSERT_EQ(static_cast<uint8_t>((expectBits >> 8) & 0xFF), dstGm[1]);
    ASSERT_EQ(static_cast<uint8_t>((expectBits >> 16) & 0xFF), dstGm[2]);
    ASSERT_EQ(static_cast<uint8_t>((expectBits >> 24) & 0xFF), dstGm[3]);
}

TEST_P(AtomicOpTestsuite_float, AtomicOpVfCallTestCase_float)
{
    auto param = GetParam();
    float value = param.value;
    float compare = param.compare;
    int num = param.num;
    int mode = param.mode;

    int32_t fpByteSize = 4;

    uint8_t dstGm[fpByteSize] = {0};
    KernelAtomic<float> op;
    op.VfCallProcess3((__gm__ float*)dstGm, value, compare, num, mode);

    float expectValue = 0.0f;
    for (int i = 0; i < num; i += 1) {
        if (mode == 0) {
            expectValue = expectValue + value;
        } else if (mode == 1) {
            expectValue = expectValue - value;
        } else if (mode == 3) {
            expectValue = expectValue > value ? expectValue : value;
        } else if (mode == 4) {
            expectValue = expectValue < value ? expectValue : value;
        } else if (mode == 2) {
            expectValue = value;
        } else if (mode == 7) {
            expectValue = expectValue == compare ? value : expectValue;
        }
    }

    // uint32_t expectBits = *reinterpret_cast<uint32_t*>(&expectValue);
    uint32_t expectBits;
    memcpy(&expectBits, &expectValue, sizeof(expectBits));
    ASSERT_EQ(static_cast<uint8_t>(expectBits & 0xFF), dstGm[0]);
    ASSERT_EQ(static_cast<uint8_t>((expectBits >> 8) & 0xFF), dstGm[1]);
    ASSERT_EQ(static_cast<uint8_t>((expectBits >> 16) & 0xFF), dstGm[2]);
    ASSERT_EQ(static_cast<uint8_t>((expectBits >> 24) & 0xFF), dstGm[3]);
}
// ================================ Test float end ==================================