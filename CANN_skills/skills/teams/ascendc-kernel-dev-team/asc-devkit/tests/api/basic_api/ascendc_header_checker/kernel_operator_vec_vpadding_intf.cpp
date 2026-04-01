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
#include "kernel_operator_vec_vpadding_intf.h"
#endif

// __aicore__ inline void VectorPadding(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint8_t padMode, const bool padSide, const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestVectorPadding1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    uint8_t padMode = 1;
    bool padSide = true;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::VectorPadding(dst, src, padMode, padSide, mask, repeatTime, repeatParams);
}

// __aicore__ inline void VectorPadding(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint8_t padMode, const bool padSide, const uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestVectorPadding2() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    uint8_t padMode = 1;
    bool padSide = true;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::VectorPadding(dst, src, padMode, padSide, mask, repeatTime, repeatParams);
}

// __aicore__ inline void VectorPadding(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint8_t padMode, const bool padSide, const uint32_t count);
extern "C" __global__ __aicore__ void KernelTestVectorPadding3() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    uint8_t padMode = 1;
    bool padSide = true;
    uint32_t count = 0;
    AscendC::VectorPadding(dst, src, padMode, padSide, count);
}
