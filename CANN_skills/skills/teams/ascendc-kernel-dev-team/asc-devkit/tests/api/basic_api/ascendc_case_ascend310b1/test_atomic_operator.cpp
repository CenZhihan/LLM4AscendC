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
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class TEST_ATOMIC_OPERATOR : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(2);
    }
    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

template<typename T, bool isAtomicAdd>
class KernelAtomic {
public:
    __aicore__ inline KernelAtomic() {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst)
    {
        srcGm.SetGlobalBuffer((__gm__ T*)src, 256);
        dstGm.SetGlobalBuffer((__gm__ T*)dst, 256);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, 256 * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, 256 * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        if constexpr(isAtomicAdd) {
            ComputeAtomicAdd();
        } else {
            ComputeGetSetAtomicConfig();
        }
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {}
    __aicore__ inline void ComputeAtomicAdd() {
        LocalTensor<T> dstLocal = outQueueZ.AllocTensor<T>();
        SetAtomicNone();
        Duplicate(dstLocal, static_cast<T>(1), 256);
        outQueueZ.EnQue<T>(dstLocal);
    }
    __aicore__ inline void ComputeGetSetAtomicConfig() {
        LocalTensor<T> dstLocal = outQueueZ.AllocTensor<T>();
        if constexpr(IsSameType<T, float>::value) {
            set_st_atomic_cfg(atomic_type_t::ATOMIC_F32, atomic_op_t::ATOMIC_SUM);
        } else if constexpr(IsSameType<T, half>::value) {
            set_st_atomic_cfg(atomic_type_t::ATOMIC_F16, atomic_op_t::ATOMIC_SUM);
        } else if constexpr(IsSameType<T, int16_t>::value) {
            set_st_atomic_cfg(atomic_type_t::ATOMIC_S16, atomic_op_t::ATOMIC_SUM);
        } else if constexpr(IsSameType<T, int32_t>::value) {
            set_st_atomic_cfg(atomic_type_t::ATOMIC_S32, atomic_op_t::ATOMIC_SUM);
        }
        pipe_barrier(PIPE_ALL);
        Duplicate(dstLocal, static_cast<T>(1), 256);
        pipe_barrier(PIPE_ALL);
        uint16_t type = 0;
        uint16_t op = 0;
        GetStoreAtomicConfig(type, op);
        pipe_barrier(PIPE_ALL);
        dstLocal.SetValue(0, type);
        dstLocal.SetValue(1, op);
        outQueueZ.EnQue<T>(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = outQueueZ.DeQue<T>();
        DataCopy(dstGm, dstLocal, 256);
        outQueueZ.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<T> srcGm;
    GlobalTensor<T> dstGm;
};

#define KERNEL_ATOMIC(type, isAtomicAdd)                                                  \
    TEST_F(TEST_ATOMIC_OPERATOR, atomic_kernel_##type##_##isAtomicAdd##_case)             \
    {                                                                                       \
        uint32_t dataSize = 256;                                                           \
        uint8_t srcGm[dataSize * sizeof(type)];                                            \
        uint8_t dstGm[dataSize * sizeof(type)];                                            \
        KernelAtomic<type, isAtomicAdd> op;                                               \
        op.Init(srcGm, dstGm);                                                              \
        op.Process();                                                                       \
        for (uint32_t i = 0; i < dataSize; i++) {                                          \
            EXPECT_EQ(dstGm[i], 0x00);                                                      \
        }                                                                                   \
    }


KERNEL_ATOMIC(float, true)
KERNEL_ATOMIC(half, true)
KERNEL_ATOMIC(int16_t, true)
KERNEL_ATOMIC(int32_t, true)

KERNEL_ATOMIC(float, false)
KERNEL_ATOMIC(half, false)
KERNEL_ATOMIC(int16_t, false)
KERNEL_ATOMIC(int32_t, false)