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
#include "kernel_operator_vec_transpose_intf.h"
#endif

#if __NPU_ARCH__ != 3102
// template <typename T> __aicore__ inline void Transpose(const LocalTensor<T>& dst, const LocalTensor<T>& src);
extern "C" __global__ __aicore__ void KernelTestTranspose1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    AscendC::Transpose(dst, src);
}
#endif

// __aicore__ inline __check_sync_alias__ void TransDataTo5HD(const LocalTensor<T> (&dstList)[NCHW_CONV_ADDR_LIST_SIZE], const LocalTensor<T> (&srcList)[NCHW_CONV_ADDR_LIST_SIZE], const TransDataTo5HDParams& nchwconvParams);
extern "C" __global__ __aicore__ void KernelTestTransDataTo5HD1() {
    AscendC::LocalTensor<float> dstList[16];
    AscendC::LocalTensor<float> srcList[16];
    AscendC::TransDataTo5HDParams nchwconvParams;
    AscendC::TransDataTo5HD(dstList, srcList, nchwconvParams);
}

// __aicore__ inline __check_sync_alias__ void TransDataTo5HD(uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE], uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE], const TransDataTo5HDParams& nchwconvParams);
extern "C" __global__ __aicore__ void KernelTestTransDataTo5HD2() {
    uint64_t dstList[16];
    uint64_t srcList[16];
    AscendC::TransDataTo5HDParams nchwconvParams;
    AscendC::TransDataTo5HD<uint32_t>(dstList, srcList, nchwconvParams);
}

#if __NPU_ARCH__ != 1001 && __NPU_ARCH__ != 3002 && __NPU_ARCH__ != 3102
// __aicore__ inline void Transpose(const LocalTensor<T> &dst, const LocalTensor<T> &src, const LocalTensor<uint8_t> &sharedTmpBuffer, const TransposeParamsExt &transposeParams);
extern "C" __global__ __aicore__ void KernelTestTranspose2() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer;
    AscendC::TransposeParamsExt transposeParams;
    AscendC::Transpose(dst, src, sharedTmpBuffer, transposeParams);
}
#endif

// __aicore__ inline __check_sync_alias__ __in_pipe__(S) __out_pipe__(V) void TransDataTo5HD(const LocalTensor<uint64_t>& dst, const LocalTensor<uint64_t>& src, const TransDataTo5HDParams& nchwconvParams);
extern "C" __global__ __aicore__ void KernelTestTransDataTo5HD3() {
    AscendC::LocalTensor<uint64_t> dst;
    AscendC::LocalTensor<uint64_t> src;
    AscendC::TransDataTo5HDParams nchwconvParams;
    AscendC::TransDataTo5HD<uint32_t>(dst, src, nchwconvParams);
}
