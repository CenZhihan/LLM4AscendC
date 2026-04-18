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

class TEST_COPY : public testing::Test {
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

template <typename Type, int32_t extent1>
struct CopyType {
    using T = Type;
    constexpr static int32_t extent = extent1;
};

template<class XType, int32_t isSetMask = 0, int32_t maskMode = 0, int32_t mode = 0, int32_t maskLen = 1, int32_t counterMask = 0>
class KernelCopy {
public:
    using X_T = typename XType::T;
    __aicore__ inline KernelCopy() {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst)
    {
        srcGm.SetGlobalBuffer((__gm__ X_T*)src, 256);
        dstGm.SetGlobalBuffer((__gm__ X_T*)dst, 256);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, 256 * sizeof(X_T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, 256 * sizeof(X_T));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        //Normal mode
        if constexpr(maskMode == 0) {
            if constexpr(mode == 0) {
                if constexpr(isSetMask == 1) {
                    ComputeNormalBitIsSetMaskTrue();
                } else {
                    ComputeNormalBitIsSetMaskFalse();
                }
            } else if constexpr(mode == 1) {
                if constexpr(isSetMask == 1) {
                    ComputeNormalContinuousIsSetMaskTrue();
                } else {
                    ComputeNormalContinuousIsSetMaskFalse();
                }
            } else {
                ASSERT(false && "Normal mode only support bit and continuous mode!!!");
            }
        } else {
            if constexpr(maskLen == 1) {
                if constexpr(isSetMask == 1) {
                    ComputeCounterMaskIsSetMaskTrue();
                } else {
                    ComputeCounterMaskIsSetMaskFalse();
                }
            } else {
                ComputeCounterMasksIsSetMaskFalse();
            }
        }
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<X_T> srcLocal = inQueueX.AllocTensor<X_T>();
        DataCopy(srcLocal, srcGm, 256);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void ComputeNormalBitIsSetMaskTrue() {
        LocalTensor<X_T> srcLocal = inQueueX.DeQue<X_T>();
        LocalTensor<X_T> dstLocal = outQueueZ.AllocTensor<X_T>();
        uint8_t repeat = sizeof(X_T);
        CopyRepeatParams repeatParams(1, 1, 8, 8);
        Duplicate(dstLocal, static_cast<X_T>(0), 256);
        pipe_barrier(PIPE_ALL);
        if (sizeof(X_T) == 2) {
            uint64_t maskList[2] = {0x5555555555555555, 0x5555555555555555};
            Copy<X_T, true>(dstLocal, srcLocal, maskList, repeat, repeatParams);
        } else {
            uint64_t maskList[2] = {0x5555555555555555, 0};
            Copy<X_T, true>(dstLocal, srcLocal, maskList, repeat, repeatParams);
        }
        outQueueZ.EnQue<X_T>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void ComputeNormalBitIsSetMaskFalse() {
        LocalTensor<X_T> srcLocal = inQueueX.DeQue<X_T>();
        LocalTensor<X_T> dstLocal = outQueueZ.AllocTensor<X_T>();
        uint8_t repeat = sizeof(X_T);
        CopyRepeatParams repeatParams(1, 1, 8, 8);
        Duplicate(dstLocal, static_cast<X_T>(0), 256);
        pipe_barrier(PIPE_ALL);
        uint64_t maskList[2] = {0, 0};
        SetMaskNorm();
        if (sizeof(X_T) == 2) {
            SetVectorMask<X_T, MaskMode::NORMAL>(0x5555555555555555, 0x5555555555555555);
        } else {
            SetVectorMask<X_T, MaskMode::NORMAL>(0, 0x5555555555555555);
        }
        Copy<X_T, false>(dstLocal, srcLocal, maskList, repeat, repeatParams);
        outQueueZ.EnQue<X_T>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void ComputeNormalContinuousIsSetMaskTrue() {
        LocalTensor<X_T> srcLocal = inQueueX.DeQue<X_T>();
        LocalTensor<X_T> dstLocal = outQueueZ.AllocTensor<X_T>();
        uint8_t repeat = sizeof(X_T);
        CopyRepeatParams repeatParams(1, 1, 8, 8);
        Duplicate(dstLocal, static_cast<X_T>(0), 256);
        pipe_barrier(PIPE_ALL);
        Copy<X_T, true>(dstLocal, srcLocal, count_mask, repeat, repeatParams);
        outQueueZ.EnQue<X_T>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void ComputeNormalContinuousIsSetMaskFalse() {
        LocalTensor<X_T> srcLocal = inQueueX.DeQue<X_T>();
        LocalTensor<X_T> dstLocal = outQueueZ.AllocTensor<X_T>();
        uint8_t repeat = sizeof(X_T);
        int32_t mask = 0;
        CopyRepeatParams repeatParams(1, 1, 8, 8);
        Duplicate(dstLocal, static_cast<X_T>(0), 256);
        pipe_barrier(PIPE_ALL);
        SetMaskNorm();
        SetVectorMask<X_T, MaskMode::NORMAL>(count_mask);
        Copy<X_T, false>(dstLocal, srcLocal, mask, repeat, repeatParams);
        outQueueZ.EnQue<X_T>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void ComputeCounterMaskIsSetMaskTrue() {
        LocalTensor<X_T> srcLocal = inQueueX.DeQue<X_T>();
        LocalTensor<X_T> dstLocal = outQueueZ.AllocTensor<X_T>();
        uint8_t repeat = sizeof(X_T);
        CopyRepeatParams repeatParams(1, 1, 8, 8);
        Duplicate(dstLocal, static_cast<X_T>(0), 256);
        pipe_barrier(PIPE_ALL);
        Copy<X_T, true>(dstLocal, srcLocal, count_mask, repeat, repeatParams);
        outQueueZ.EnQue<X_T>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void ComputeCounterMaskIsSetMaskFalse() {
        LocalTensor<X_T> srcLocal = inQueueX.DeQue<X_T>();
        LocalTensor<X_T> dstLocal = outQueueZ.AllocTensor<X_T>();
        uint8_t repeat = sizeof(X_T);
        int32_t mask = 0;
        CopyRepeatParams repeatParams(1, 1, 8, 8);
        Duplicate(dstLocal, static_cast<X_T>(0), 256);
        pipe_barrier(PIPE_ALL);
        SetMaskCount();
        SetVectorMask<X_T, MaskMode::COUNTER>(count_mask);
        Copy<X_T, false>(dstLocal, srcLocal, mask, repeat, repeatParams);
        SetMaskNorm();
        outQueueZ.EnQue<X_T>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void ComputeCounterMasksIsSetMaskFalse() {
        LocalTensor<X_T> srcLocal = inQueueX.DeQue<X_T>();
        LocalTensor<X_T> dstLocal = outQueueZ.AllocTensor<X_T>();
        uint8_t repeat = sizeof(X_T);
        int32_t mask = 0;
        CopyRepeatParams repeatParams(1, 1, 8, 8);
        Duplicate(dstLocal, static_cast<X_T>(0), 256);
        pipe_barrier(PIPE_ALL);
        SetMaskCount();
        SetVectorMask<X_T, MaskMode::COUNTER>(0, count_mask);
        Copy<X_T, false>(dstLocal, srcLocal, mask, repeat, repeatParams);
        SetMaskNorm();
        outQueueZ.EnQue<X_T>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<X_T> dstLocal = outQueueZ.DeQue<X_T>();
        DataCopy(dstGm, dstLocal, 256);
        outQueueZ.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<X_T> srcGm;
    GlobalTensor<X_T> dstGm;
    int32_t count_mask{counterMask};
};

#define KERNEL_COPY_M310(dataType, isSetMask, maskMode, mode, maskLen, counterMask)         \
    TEST_F(TEST_COPY, copy_kernel_##dataType##_is_set_mask_##isSetMask##_mask_mode_##maskMode##_##mode##_mask_len_##maskLen##_##counterMask##_case) \
    {                                                                                       \
        uint32_t dataSize = 256;                                                           \
        uint8_t srcGm[dataSize * sizeof(dataType)];                                        \
        uint8_t dstGm[dataSize * sizeof(dataType)];                                        \
        typedef CopyType<dataType, 256> xType;                                              \
        KernelCopy<xType, isSetMask, maskMode, mode, maskLen, counterMask> op;              \
        op.Init(srcGm, dstGm);                                                              \
        op.Process();                                                                       \
        dataType golden[dataSize];                                                         \
        uint8_t repeat = sizeof(dataType);                                                  \
        uint8_t vecLen = 256 / sizeof(dataType);                                            \
        if (maskMode == 0 && mode == 0) {                                                   \
            for (int i = 0; i < repeat; i++) {                                              \
                for (int j = 0; j < vecLen; j++) {                                          \
                    if (j % 2 == 0) {                                                       \
                        golden[i*vecLen + j] = srcGm[i*vecLen + j];                         \
                    } else {                                                                \
                        golden[i*vecLen + j] = 0;                                           \
                    }                                                                       \
                }                                                                           \
            }                                                                               \
        } else {                                                                            \
            if (isSetMask == 1) {                                                           \
                for (int r = 0; r < repeat; r++) {                                          \
                    for (int i = 0; i < vecLen; i++) {                                      \
                        if (i < counterMask) {                                              \
                            golden[r*vecLen + i] = srcGm[r*vecLen + i];                     \
                        } else {                                                            \
                            golden[r*vecLen + i] = 0;                                       \
                        }                                                                   \
                    }                                                                       \
                }                                                                           \
            } else {                                                                        \
                for (int i = 0; i < dataSize; i++) {                                       \
                    if (i < counterMask) {                                                  \
                        golden[i] = srcGm[i];                                               \
                    } else {                                                                \
                        golden[i] = 0;                                                      \
                    }                                                                       \
                }                                                                           \
            }                                                                               \
        }                                                                                   \
        for (uint32_t i = 0; i < dataSize; i++) {                                          \
            EXPECT_EQ(dstGm[i], 0x00);                                                      \
        }                                                                                   \
    }

KERNEL_COPY_M310(int16_t, 0, 0, 0, 2, 0)
KERNEL_COPY_M310(uint16_t, 0, 0, 0, 2, 0)
KERNEL_COPY_M310(int32_t, 0, 0, 0, 2, 0)
KERNEL_COPY_M310(uint32_t, 0, 0, 0, 2, 0)
KERNEL_COPY_M310(float, 0, 0, 0, 2, 0)
KERNEL_COPY_M310(half, 0, 0, 0, 2, 0)

KERNEL_COPY_M310(int16_t, 1, 0, 0, 2, 0)
KERNEL_COPY_M310(uint16_t, 1, 0, 0, 2, 0)
KERNEL_COPY_M310(int32_t, 1, 0, 0, 2, 0)
KERNEL_COPY_M310(uint32_t, 1, 0, 0, 2, 0)
KERNEL_COPY_M310(float, 1, 0, 0, 2, 0)
KERNEL_COPY_M310(half, 1, 0, 0, 2, 0)

KERNEL_COPY_M310(int16_t, 0, 0, 1, 1, 40)
KERNEL_COPY_M310(uint16_t, 0, 0, 1, 1, 100)
KERNEL_COPY_M310(int32_t, 0, 0, 1, 1, 121)
KERNEL_COPY_M310(uint32_t, 0, 0, 1, 1, 200)
KERNEL_COPY_M310(float, 0, 0, 1, 1, 158)
KERNEL_COPY_M310(half, 0, 0, 1, 1, 39)

KERNEL_COPY_M310(int16_t, 1, 0, 1, 1, 80)
KERNEL_COPY_M310(uint16_t, 1, 0, 1, 1, 123)
KERNEL_COPY_M310(int32_t, 1, 0, 1, 1, 61)
KERNEL_COPY_M310(uint32_t, 1, 0, 1, 1, 63)
KERNEL_COPY_M310(float, 1, 0, 1, 1, 50)
KERNEL_COPY_M310(half, 1, 0, 1, 1, 68)

KERNEL_COPY_M310(int16_t, 0, 1, 1, 1, 92)
KERNEL_COPY_M310(uint16_t, 0, 1, 1, 1, 89)
KERNEL_COPY_M310(int32_t, 0, 1, 1, 1, 249)
KERNEL_COPY_M310(uint32_t, 0, 1, 1, 1, 201)
KERNEL_COPY_M310(float, 0, 1, 1, 1, 157)
KERNEL_COPY_M310(half, 0, 1, 1, 1, 132)

KERNEL_COPY_M310(int16_t, 1, 0, 1, 1, 123)
KERNEL_COPY_M310(uint16_t, 1, 0, 1, 1, 104)
KERNEL_COPY_M310(int32_t, 1, 0, 1, 1, 48)
KERNEL_COPY_M310(uint32_t, 1, 0, 1, 1, 32)
KERNEL_COPY_M310(float, 1, 0, 1, 1, 55)
KERNEL_COPY_M310(half, 1, 0, 1, 1, 101)

KERNEL_COPY_M310(int16_t, 0, 1, 1, 2, 199)
KERNEL_COPY_M310(uint16_t, 0, 1, 1, 2, 187)
KERNEL_COPY_M310(int32_t, 0, 1, 1, 2, 136)
KERNEL_COPY_M310(uint32_t, 0, 1, 1, 2, 144)
KERNEL_COPY_M310(float, 0, 1, 1, 2, 243)
KERNEL_COPY_M310(half, 0, 1, 1, 2, 205)
