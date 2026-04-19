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
#include "kernel_operator_conv2d_intf.h"
#endif

// __aicore__ inline __in_pipe__(MTE2)__out_pipe__(MTE3) void Conv2D(const LocalTensor<T> &dst, const LocalTensor<T> &bias, const LocalTensor<U> &featureMap, const LocalTensor<U> &weight, Conv2dParams &conv2dParams, Conv2dTilling &tilling);
extern "C" __global__ __aicore__ void KernelTestConv2D1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> bias;
    AscendC::LocalTensor<half> featureMap;
    AscendC::LocalTensor<half> weight;
    AscendC::Conv2dParams conv2dParams;
    AscendC::Conv2dTilling tilling;
    AscendC::Conv2D(dst, bias, featureMap, weight, conv2dParams, tilling);
}
