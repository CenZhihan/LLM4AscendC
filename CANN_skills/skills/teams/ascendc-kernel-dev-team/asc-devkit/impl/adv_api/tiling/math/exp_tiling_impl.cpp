/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file exp_tiling_impl.cpp
 * \brief
 */
#include <cstdint>
#include "include/adv_api/math/exp_tiling.h"
#include "graph/tensor.h"

namespace AscendC {
namespace {
constexpr uint32_t EXP_HALF_CALC_PROC = 8;
constexpr uint32_t EXP_FLOAT_CALC_PROC = 3;
constexpr uint32_t EXP_ONE_REPEAT_BYTE_SIZE = 256;
constexpr uint32_t EXP_TWO_TIMES = 2;                  // normal constant 2
constexpr uint32_t EXP_THREE_TIMES = 3;                // normal constant 3
constexpr uint32_t EXP_FOUR_TIMES = 4;                 // normal constant 4

inline uint32_t GetExpMaxTmpSize(const uint32_t inputSize, const uint32_t typeSize,
    const bool isReuseSource)
{
    // FP32 tmpBuffer (must be at least 256 Bytes)
    const uint32_t tmpBufferSize = std::max<uint64_t>(inputSize * sizeof(float), EXP_ONE_REPEAT_BYTE_SIZE);
    // high precision
    uint32_t numberOfTmpBuf = EXP_FOUR_TIMES;
    if (typeSize == sizeof(float)) { // FP32
        numberOfTmpBuf = isReuseSource ? EXP_TWO_TIMES : EXP_THREE_TIMES;
    }
    return numberOfTmpBuf * tmpBufferSize;
}

inline uint32_t GetExpMinTmpSize(const uint32_t typeSize, const bool isReuseSource)
{
    // high precision
    uint32_t numberOfTmpBuf = EXP_FOUR_TIMES;
    if (typeSize == sizeof(float)) { // FP32
        numberOfTmpBuf = isReuseSource ? EXP_TWO_TIMES : EXP_THREE_TIMES;
    }
    return numberOfTmpBuf * EXP_ONE_REPEAT_BYTE_SIZE;
}
} // namespace

// input: sizeof(dtype) * size      buffer needed: x * size
// factor = x / sizeof(dtype)       if buffer needed is x * size + extra, then extra need to be updated
void GetExpTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuffer)
{
    // FP16 input: 2 * size, buffer: 4 FP32 tensor = 16 * size, factor = 16 / 2 = 8
    // FP32 input: 4 * size, buffer: 3 FP32 tensor = 12 * size, factor = 12 / 4 = 3
    extraBuffer = 0;
    maxLiveNodeCount = (typeSize == sizeof(float)) ? EXP_FLOAT_CALC_PROC : EXP_HALF_CALC_PROC;
}

bool GetExpMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    const uint32_t inputSize = srcShape.GetShapeSize();
    minValue = GetExpMinTmpSize(typeSize, isReuseSource);
    maxValue = GetExpMaxTmpSize(inputSize, typeSize, isReuseSource);
    return true;
}

} // namespace AscendC
