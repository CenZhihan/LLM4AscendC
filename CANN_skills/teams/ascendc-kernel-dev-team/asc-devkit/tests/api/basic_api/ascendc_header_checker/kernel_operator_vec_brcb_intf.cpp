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
#include "kernel_operator_vec_brcb_intf.h"
#endif

#if __NPU_ARCH__ != 3102
// __aicore__ inline void Brcb(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const uint8_t repeatTime, const BrcbRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestBrcb1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src0;
    uint8_t repeatTime = 1;
    AscendC::BrcbRepeatParams repeatParams;
    AscendC::Brcb(dst, src0, repeatTime, repeatParams);
}
#endif
