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
#include "kernel_operator_vec_binary_scalar_intf.h"
#endif

// __aicore__ inline void Adds(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestAdds1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Adds(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Adds(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestAdds2() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Adds(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Adds(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestAdds3() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Adds(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Adds(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestAdds4() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Adds(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Adds(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestAdds7() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    int32_t count = 0;
    AscendC::Adds(dst, src, scalarValue, count);
}

// __aicore__ inline void Adds(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestAdds8() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    int32_t count = 0;
    AscendC::Adds(dst, src, scalarValue, count);
}

// __aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestMuls1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Muls(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestMuls2() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Muls(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestMuls3() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Muls(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestMuls4() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Muls(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestMuls7() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    int32_t count = 0;
    AscendC::Muls(dst, src, scalarValue, count);
}

// __aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestMuls8() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    int32_t count = 0;
    AscendC::Muls(dst, src, scalarValue, count);
}

// __aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestMaxs1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Maxs(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestMaxs2() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Maxs(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestMaxs3() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Maxs(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestMaxs4() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Maxs(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestMaxs7() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    int32_t count = 0;
    AscendC::Maxs(dst, src, scalarValue, count);
}

// __aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestMaxs8() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    int32_t count = 0;
    AscendC::Maxs(dst, src, scalarValue, count);
}

// __aicore__ inline void Mins(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestMins1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Mins(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Mins(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestMins2() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Mins(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Mins(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestMins3() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Mins(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Mins(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestMins4() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::Mins(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void Mins(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestMins7() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    int32_t count = 0;
    AscendC::Mins(dst, src, scalarValue, count);
}

// __aicore__ inline void Mins(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestMins8() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    int32_t count = 0;
    AscendC::Mins(dst, src, scalarValue, count);
}

// __aicore__ inline void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestShiftLeft1() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    uint32_t scalarValue = 0;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::ShiftLeft(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestShiftLeft2() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    uint32_t scalarValue = 0;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::ShiftLeft(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestShiftLeft3() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    uint32_t scalarValue = 0;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::ShiftLeft(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestShiftLeft4() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    uint32_t scalarValue = 0;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::ShiftLeft(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestShiftLeft5() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    uint32_t scalarValue = 0;
    int32_t count = 0;
    AscendC::ShiftLeft(dst, src, scalarValue, count);
}

// __aicore__ inline void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestShiftLeft6() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    uint32_t scalarValue = 0;
    int32_t count = 0;
    AscendC::ShiftLeft(dst, src, scalarValue, count);
}

// __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn = false);
extern "C" __global__ __aicore__ void KernelTestShiftRight1() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    uint32_t scalarValue = 0;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    bool roundEn = false;
    AscendC::ShiftRight(dst, src, scalarValue, mask, repeatTime, repeatParams, false);
}

// __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn);
extern "C" __global__ __aicore__ void KernelTestShiftRight2() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    uint32_t scalarValue = 0;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    bool roundEn = true;
    AscendC::ShiftRight(dst, src, scalarValue, mask, repeatTime, repeatParams, roundEn);
}

// __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn = false);
extern "C" __global__ __aicore__ void KernelTestShiftRight3() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    uint32_t scalarValue = 0;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    bool roundEn = false;
    AscendC::ShiftRight(dst, src, scalarValue, mask, repeatTime, repeatParams, false);
}

// __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn);
extern "C" __global__ __aicore__ void KernelTestShiftRight4() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    uint32_t scalarValue = 0;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    bool roundEn = true;
    AscendC::ShiftRight(dst, src, scalarValue, mask, repeatTime, repeatParams, roundEn);
}

// __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestShiftRight5() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    uint32_t scalarValue = 0;
    int32_t count = 0;
    AscendC::ShiftRight(dst, src, scalarValue, count);
}

// __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestShiftRight6() {
    AscendC::LocalTensor<uint32_t> dst;
    AscendC::LocalTensor<uint32_t> src;
    uint32_t scalarValue = 0;
    int32_t count = 0;
    AscendC::ShiftRight(dst, src, scalarValue, count);
}

// __aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestLeakyRelu1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::LeakyRelu(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestLeakyRelu2() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask[2] = {0};
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::LeakyRelu(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestLeakyRelu3() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::LeakyRelu(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
extern "C" __global__ __aicore__ void KernelTestLeakyRelu4() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint8_t repeatTime = 1;
    AscendC::UnaryRepeatParams repeatParams;
    AscendC::LeakyRelu(dst, src, scalarValue, mask, repeatTime, repeatParams);
}

// __aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestLeakyRelu5() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    int32_t count = 0;
    AscendC::LeakyRelu(dst, src, scalarValue, count);
}

// __aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, const int32_t& count);
extern "C" __global__ __aicore__ void KernelTestLeakyRelu6() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    float scalarValue = 0.0f;
    int32_t count = 0;
    AscendC::LeakyRelu(dst, src, scalarValue, count);
}
