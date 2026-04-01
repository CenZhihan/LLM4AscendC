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
#include "kernel_operator_vec_gather_intf.h"
#endif

#if __NPU_ARCH__ != 3102
// __aicore__ inline void Gatherb(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<uint32_t>& offset, const uint8_t repeatTime, const GatherRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestGatherb1() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src0;
    AscendC::LocalTensor<uint32_t> offset;
    uint8_t repeatTime = 1;
    AscendC::GatherRepeatParams repeatParams;
    AscendC::Gatherb(dst, src0, offset, repeatTime, repeatParams);
}

// __aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseAddr, const uint64_t mask, const uint8_t repeatTime, const uint16_t dstRepStride);
extern "C" __global__ __aicore__ void KernelTestGather1() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    AscendC::LocalTensor<uint32_t> srcOffset;
    uint32_t srcBaseAddr = 0;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    uint16_t dstRepStride = 1;
    AscendC::Gather(dst, src, srcOffset, srcBaseAddr, mask, repeatTime, dstRepStride);
}

// __aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseAddr, const uint64_t mask[], const uint8_t repeatTime, const uint16_t dstRepStride);
extern "C" __global__ __aicore__ void KernelTestGather2() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    AscendC::LocalTensor<uint32_t> srcOffset;
    uint32_t srcBaseAddr = 0;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    uint16_t dstRepStride = 1;
    AscendC::Gather(dst, src, srcOffset, srcBaseAddr, mask, repeatTime, dstRepStride);
}

// __aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseAddr, const uint32_t count);
extern "C" __global__ __aicore__ void KernelTestGather3() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    AscendC::LocalTensor<uint32_t> srcOffset;
    uint32_t srcBaseAddr = 0;
    uint32_t count = 0;
    AscendC::Gather(dst, src, srcOffset, srcBaseAddr, count);
}
#endif
