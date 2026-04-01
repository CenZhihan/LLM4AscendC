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
#include "kernel_operator_set_atomic_intf.h"
#endif

// __aicore__ inline void SetAtomicType();
extern "C" __global__ __aicore__ void KernelTestSetAtomicType1() {
    AscendC::SetAtomicType<int32_t>();
}

// __aicore__ inline void SetAtomicAdd();
extern "C" __global__ __aicore__ void KernelTestSetAtomicAdd1() {
    AscendC::SetAtomicAdd<int32_t>();
}

// __aicore__ inline void SetAtomicNone();
extern "C" __global__ __aicore__ void KernelTestSetAtomicNone1() {
    AscendC::SetAtomicNone();
}

// __aicore__ inline void SetAtomicMax();
extern "C" __global__ __aicore__ void KernelTestSetAtomicMax1() {
    AscendC::SetAtomicMax<int32_t>();
}

// __aicore__ inline void SetAtomicMin();
extern "C" __global__ __aicore__ void KernelTestSetAtomicMin1() {
    AscendC::SetAtomicMin<int32_t>();
}
