/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef VERIFY_SINGLE_HEADER
#include "kernel_operator.h"
#else
#include "kernel_operator_vec_createvecindex_intf.h"
#endif

// __aicore__ inline __in_pipe__(S) __out_pipe__(V) void CreateVecIndex(LocalTensor<T> &dst, const T &firstValue, uint64_t mask, uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride);
extern "C" __global__ __aicore__ void KernelTestCreateVecIndex1() {
    AscendC::LocalTensor<float> dst;
    float firstValue = 0.0f;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    uint16_t dstBlkStride = 1;
    uint8_t dstRepStride = 1;
    AscendC::CreateVecIndex(dst, firstValue, mask, repeatTime, dstBlkStride, dstRepStride);
}

// __aicore__ inline __in_pipe__(S) __out_pipe__(V) void CreateVecIndex(LocalTensor<T> &dst, const T &firstValue, uint64_t mask[], uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride);
extern "C" __global__ __aicore__ void KernelTestCreateVecIndex2() {
    AscendC::LocalTensor<float> dst;
    float firstValue = 0.0f;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    uint16_t dstBlkStride = 1;
    uint8_t dstRepStride = 1;
    AscendC::CreateVecIndex(dst, firstValue, mask, repeatTime, dstBlkStride, dstRepStride);
}

// __aicore__ inline __in_pipe__(S) __out_pipe__(V) void CreateVecIndex(LocalTensor<T> dst, const T &firstValue, uint32_t count);
extern "C" __global__ __aicore__ void KernelTestCreateVecIndex3() {
    AscendC::LocalTensor<float> dst;
    float firstValue = 0.0f;
    uint32_t count = 0;
    AscendC::CreateVecIndex(dst, firstValue, count);
}
