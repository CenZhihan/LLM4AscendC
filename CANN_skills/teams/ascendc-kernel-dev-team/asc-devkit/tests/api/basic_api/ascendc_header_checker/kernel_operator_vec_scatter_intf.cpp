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
#include "kernel_operator_vec_scatter_intf.h"
#endif

// __aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& dstOffset, const uint32_t dstBaseAddr, const uint64_t mask, const uint8_t repeatTime, const uint8_t srcRepStride);
extern "C" __global__ __aicore__ void KernelTestScatter1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    AscendC::LocalTensor<uint32_t> dstOffset;
    uint32_t dstBaseAddr = 0;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    uint8_t srcRepStride = 1;
    AscendC::Scatter(dst, src, dstOffset, dstBaseAddr, mask, repeatTime, srcRepStride);
}

// __aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& dstOffset, const uint32_t dstBaseAddr, const uint64_t mask[], const uint8_t repeatTime, const uint8_t srcRepStride);
extern "C" __global__ __aicore__ void KernelTestScatter2() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    AscendC::LocalTensor<uint32_t> dstOffset;
    uint32_t dstBaseAddr = 0;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    uint8_t srcRepStride = 1;
    AscendC::Scatter(dst, src, dstOffset, dstBaseAddr, mask, repeatTime, srcRepStride);
}

#if __NPU_ARCH__ != 3102
// __aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& dstOffset, const uint32_t dstBaseAddr, const uint32_t count);
extern "C" __global__ __aicore__ void KernelTestScatter3() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    AscendC::LocalTensor<uint32_t> dstOffset;
    uint32_t dstBaseAddr = 0;
    uint32_t count = 0;
    AscendC::Scatter(dst, src, dstOffset, dstBaseAddr, count);
}
#endif
