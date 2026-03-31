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
#include "kernel_operator_vec_ternary_scalar_intf.h"
#endif

// __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestAxpy1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Axpy(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestAxpy2() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Axpy(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestAxpy3() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    int32_t count = 0;
    AscendC::Axpy(dst, src, scalarValue, count);
}
