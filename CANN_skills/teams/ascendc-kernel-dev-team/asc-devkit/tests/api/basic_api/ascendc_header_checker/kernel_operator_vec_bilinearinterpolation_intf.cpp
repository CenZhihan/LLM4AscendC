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
#include "kernel_operator_vec_bilinearinterpolation_intf.h"
#endif

// __aicore__ inline void BilinearInterpolation(const LocalTensor<T> &dst, const LocalTensor<T> &src0, const LocalTensor<uint32_t> &src0Offset, const LocalTensor<T> &src1, uint64_t mask, uint8_t hRepeat, bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat, const LocalTensor<uint8_t> &sharedTmpBuffer);
extern "C" __global__ __aicore__ void KernelTestBilinearInterpolation1() {
    AscendC::LocalTensor<half> dst;
    AscendC::LocalTensor<half> src0;
    AscendC::LocalTensor<uint32_t> src0Offset;
    AscendC::LocalTensor<half> src1;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t hRepeat = 1;
    bool repeatMode = true;
    uint16_t dstBlkStride = 1;
    uint16_t vROffset = 1;
    uint8_t vRepeat = 1;
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer;
    AscendC::BilinearInterpolation(dst, src0, src0Offset, src1, mask, hRepeat, repeatMode, dstBlkStride, vROffset, vRepeat, sharedTmpBuffer);
}

#if __NPU_ARCH__ != 1001 && __NPU_ARCH__ != 3002 && __NPU_ARCH__ != 3102
// __aicore__ inline void BilinearInterpolation(const LocalTensor<T> &dst, const LocalTensor<T> &src0, const LocalTensor<uint32_t> &src0Offset, const LocalTensor<T> &src1, uint64_t mask[], uint8_t hRepeat, bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat, const LocalTensor<uint8_t> &sharedTmpBuffer);
extern "C" __global__ __aicore__ void KernelTestBilinearInterpolation2() {
    AscendC::LocalTensor<half> dst;
    AscendC::LocalTensor<half> src0;
    AscendC::LocalTensor<uint32_t> src0Offset;
    AscendC::LocalTensor<half> src1;
    uint64_t mask[2] = {0};
    uint8_t hRepeat = 1;
    bool repeatMode = true;
    uint16_t dstBlkStride = 1;
    uint16_t vROffset = 1;
    uint8_t vRepeat = 1;
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer;
    AscendC::BilinearInterpolation(dst, src0, src0Offset, src1, mask, hRepeat, repeatMode, dstBlkStride, vROffset, vRepeat, sharedTmpBuffer);
}
#endif
