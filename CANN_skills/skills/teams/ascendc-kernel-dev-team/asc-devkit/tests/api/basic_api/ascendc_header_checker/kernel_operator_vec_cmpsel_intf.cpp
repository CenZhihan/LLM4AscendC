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
#include "kernel_operator_vec_cmpsel_intf.h"
#endif

// __aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask[], uint8_t repeatTime, const BinaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestCompare1() {
    AscendC::LocalTensor<uint8_t> dst;
    AscendC::LocalTensor<float> src0;
    AscendC::LocalTensor<float> src1;
    AscendC::CMPMODE cmpMode;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::BinaryRepeatParams repeatParams;
    AscendC::Compare(dst, src0, src1, cmpMode, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestCompare2() {
    AscendC::LocalTensor<uint8_t> dst;
    AscendC::LocalTensor<float> src0;
    AscendC::LocalTensor<float> src1;
    AscendC::CMPMODE cmpMode;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::BinaryRepeatParams repeatParams;
    AscendC::Compare(dst, src0, src1, cmpMode, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Compare(const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask[], const BinaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestCompare3() {
    AscendC::LocalTensor<float> src0;
    AscendC::LocalTensor<float> src1;
    AscendC::CMPMODE cmpMode;
    uint64_t mask[2] = {0};
    AscendC::BinaryRepeatParams repeatParams;
    AscendC::Compare(src0, src1, cmpMode, mask, repeatParams);
}

// __aicore__ inline void Compare(const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask, const BinaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestCompare4() {
    AscendC::LocalTensor<float> src0;
    AscendC::LocalTensor<float> src1;
    AscendC::CMPMODE cmpMode;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    AscendC::BinaryRepeatParams repeatParams;
    AscendC::Compare(src0, src1, cmpMode, mask, repeatParams);
}

// __aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, uint32_t count);
extern "C" __global__ __aicore__ void KernelTestCompare5() {
    AscendC::LocalTensor<uint8_t> dst;
    AscendC::LocalTensor<float> src0;
    AscendC::LocalTensor<float> src1;
    AscendC::CMPMODE cmpMode;
    uint32_t count = 0;
    AscendC::Compare(dst, src0, src1, cmpMode, count);
}

// __aicore__ inline void GetCmpMask(const LocalTensor<T>& dst);
extern "C" __global__ __aicore__ void KernelTestGetCmpMask1() {
    AscendC::LocalTensor<float> dst;
    AscendC::GetCmpMask(dst);
}

// __aicore__ inline void SetCmpMask(const LocalTensor<T>& src);
extern "C" __global__ __aicore__ void KernelTestSetCmpMask1() {
    AscendC::LocalTensor<float> src;
    AscendC::SetCmpMask(src);
}

// __aicore__ inline void CompareScalar(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const T src1Scalar, CMPMODE cmpMode, const uint64_t mask[], uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestCompareScalar1() {
    AscendC::LocalTensor<uint8_t> dst;
    AscendC::LocalTensor<float> src0;
    float src1Scalar = 0.0f;
    AscendC::CMPMODE cmpMode;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::CompareScalar(dst, src0, src1Scalar, cmpMode, mask, repeatTime, repeatParams);
}

// __aicore__ inline void CompareScalar(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const T src1Scalar, CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestCompareScalar2() {
    AscendC::LocalTensor<uint8_t> dst;
    AscendC::LocalTensor<float> src0;
    float src1Scalar = 0.0f;
    AscendC::CMPMODE cmpMode;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::CompareScalar(dst, src0, src1Scalar, cmpMode, mask, repeatTime, repeatParams);
}

// __aicore__ inline void CompareScalar(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const T src1Scalar, CMPMODE cmpMode, uint32_t count);
extern "C" __global__ __aicore__ void KernelTestCompareScalar3() {
    AscendC::LocalTensor<uint8_t> dst;
    AscendC::LocalTensor<float> src0;
    float src1Scalar = 0.0f;
    AscendC::CMPMODE cmpMode;
    uint32_t count = 0;
    AscendC::CompareScalar(dst, src0, src1Scalar, cmpMode, count);
}

// __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask, const LocalTensor<T>& src0, const LocalTensor<T>& src1, SELMODE selMode, uint64_t mask[], uint8_t repeatTime, const BinaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestSelect1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<uint8_t> selMask;
    AscendC::LocalTensor<float> src0;
    AscendC::LocalTensor<float> src1;
    AscendC::SELMODE selMode;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::BinaryRepeatParams repeatParams;
    AscendC::Select(dst, selMask, src0, src1, selMode, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask, const LocalTensor<T>& src0, const LocalTensor<T>& src1, SELMODE selMode, uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestSelect2() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<uint8_t> selMask;
    AscendC::LocalTensor<float> src0;
    AscendC::LocalTensor<float> src1;
    AscendC::SELMODE selMode;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::BinaryRepeatParams repeatParams;
    AscendC::Select(dst, selMask, src0, src1, selMode, mask, repeatTime, repeatParams);
}

// // __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, uint8_t repeatTime, const BinaryRepeatParams& repeatParams);
// extern "C" __global__ __aicore__ void KernelTestSelect3() {
//     AscendC::LocalTensor<uint32_t> dst;
//     AscendC::LocalTensor<uint32_t> src0;
//     AscendC::LocalTensor<uint32_t> src1;
//     uint8_t repeatTime = 1;
//     AscendC::BinaryRepeatParams repeatParams;
//     AscendC::Select(dst, src0, src1, repeatTime, repeatParams);
// }

// __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask, const LocalTensor<T>& src0, uint8_t repeatTime, const BinaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestSelect4() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<uint8_t> selMask;
    AscendC::LocalTensor<float> src0;
    uint8_t repeatTime = 1;
    AscendC::BinaryRepeatParams repeatParams;
    AscendC::Select(dst, selMask, src0, repeatTime, repeatParams);
}

// __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask, const LocalTensor<T>& src0, const LocalTensor<T>& src1, SELMODE selMode, uint32_t count);
extern "C" __global__ __aicore__ void KernelTestSelect5() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<uint8_t> selMask;
    AscendC::LocalTensor<float> src0;
    AscendC::LocalTensor<float> src1;
    AscendC::SELMODE selMode;
    uint32_t count = 0;
    AscendC::Select(dst, selMask, src0, src1, selMode, count);
}

// __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask, const LocalTensor<T>& src0, T src1, SELMODE selMode, uint64_t mask[], uint8_t repeatTime, const BinaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestSelect6() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<uint8_t> selMask;
    AscendC::LocalTensor<float> src0;
    float src1 = 0.0f;
    AscendC::SELMODE selMode;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::BinaryRepeatParams repeatParams;
    AscendC::Select(dst, selMask, src0, src1, selMode, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask, const LocalTensor<T>& src0, T src1, SELMODE selMode, uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestSelect7() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<uint8_t> selMask;
    AscendC::LocalTensor<float> src0;
    float src1 = 0.0f;
    AscendC::SELMODE selMode;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::BinaryRepeatParams repeatParams;
    AscendC::Select(dst, selMask, src0, src1, selMode, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask, const LocalTensor<T>& src0, T src1, SELMODE selMode, uint32_t count);
extern "C" __global__ __aicore__ void KernelTestSelect8() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<uint8_t> selMask;
    AscendC::LocalTensor<float> src0;
    float src1 = 0.0f;
    AscendC::SELMODE selMode;
    uint32_t count = 0;
    AscendC::Select(dst, selMask, src0, src1, selMode, count);
}
