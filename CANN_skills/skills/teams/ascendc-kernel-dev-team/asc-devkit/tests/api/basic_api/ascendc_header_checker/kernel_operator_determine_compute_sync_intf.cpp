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
#include "kernel_operator_determine_compute_sync_intf.h"
#endif

// __aicore__ inline void InitDetermineComputeWorkspace(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace);
extern "C" __global__ __aicore__ void KernelTestInitDetermineComputeWorkspace1() {
    AscendC::GlobalTensor<int32_t> gmWorkspace;
    AscendC::LocalTensor<int32_t> ubWorkspace;
    AscendC::InitDetermineComputeWorkspace(gmWorkspace, ubWorkspace);
}

// __aicore__ inline void WaitPreBlock(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace);
extern "C" __global__ __aicore__ void KernelTestWaitPreBlock1() {
    AscendC::GlobalTensor<int32_t> gmWorkspace;
    AscendC::LocalTensor<int32_t> ubWorkspace;
    AscendC::WaitPreBlock(gmWorkspace, ubWorkspace);
}

// __aicore__ inline void NotifyNextBlock(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace);
extern "C" __global__ __aicore__ void KernelTestNotifyNextBlock1() {
    AscendC::GlobalTensor<int32_t> gmWorkspace;
    AscendC::LocalTensor<int32_t> ubWorkspace;
    AscendC::NotifyNextBlock(gmWorkspace, ubWorkspace);
}
