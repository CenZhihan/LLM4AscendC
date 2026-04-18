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
#include "kernel_operator_gemm_intf.h"
#endif

// __aicore__ inline __inout_pipe__(V) void Gemm(const LocalTensor<T>& dst, const LocalTensor<U>& src0, const LocalTensor<S>& src1, const uint32_t m, const uint32_t k, const uint32_t n, GemmTiling tilling, bool partialsum = true, int32_t initValue = 0);
extern "C" __global__ __aicore__ void KernelTestGemm1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<half> src0;
    AscendC::LocalTensor<half> src1;
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    AscendC::GemmTiling tilling;
    bool partialsum = true ;
    int32_t initValue = 0;
    AscendC::Gemm(dst, src0, src1, m, k, n, tilling, true, 0);
}
