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
#include "kernel_operator_vec_mulcast_intf.h"
#endif

// __aicore__ inline void MulCast(const LocalTensor<T> &dst, const LocalTensor<U> &src0, const LocalTensor<U> &src1, uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams &repeatParams);
extern "C" __global__ __aicore__ void KernelTestMulCast1() {
    AscendC::LocalTensor<int8_t> dst;
    AscendC::LocalTensor<half> src0;
    AscendC::LocalTensor<half> src1;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::BinaryRepeatParams repeatParams;
    AscendC::MulCast(dst, src0, src1, mask, repeatTime, repeatParams);
}

// __aicore__ inline void MulCast(const LocalTensor<T> &dst, const LocalTensor<U> &src0, const LocalTensor<U> &src1, uint64_t mask[], const uint8_t repeatTime, const BinaryRepeatParams &repeatParams);
extern "C" __global__ __aicore__ void KernelTestMulCast2() {
    AscendC::LocalTensor<int8_t> dst;
    AscendC::LocalTensor<half> src0;
    AscendC::LocalTensor<half> src1;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::BinaryRepeatParams repeatParams;
    AscendC::MulCast(dst, src0, src1, mask, repeatTime, repeatParams);
}

// __aicore__ inline void MulCast(const LocalTensor<T> &dst, const LocalTensor<U> &src0, const LocalTensor<U> &src1, uint32_t count);
extern "C" __global__ __aicore__ void KernelTestMulCast3() {
    AscendC::LocalTensor<int8_t> dst;
    AscendC::LocalTensor<half> src0;
    AscendC::LocalTensor<half> src1;
    uint32_t count = 0;
    AscendC::MulCast(dst, src0, src1, count);
}
