/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file log_tiling.cpp
 * \brief
 */
#include "include/adv_api/math/log_tiling.h"

#include <cstdint>

#include "graph/tensor.h"
#include "../../detail/host_log.h"
namespace AscendC {
namespace {
constexpr uint32_t LOG2_HALF_CALC_FAC = 2;
constexpr uint32_t LOG2_FLOAT_CALC_FAC = 0;
constexpr uint32_t LOG2_TMP_SIZE = 256;

inline uint32_t GetLog2MinTmpSize(const uint32_t typeSize)
{
    if (typeSize == sizeof(uint16_t)) {
        return LOG2_TMP_SIZE;
    }
    return 0;
}

inline uint32_t GetLog2MaxTmpSize(const uint32_t inputSize, const uint32_t typeSize)
{
    if (typeSize == sizeof(uint16_t)) { // half
        return std::max(inputSize * static_cast<uint32_t>(sizeof(float)), LOG2_TMP_SIZE);
    }
    return 0;
}
} // namespace

void GetLogMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    (void)srcShape;
    (void)typeSize;
    (void)isReuseSource;
    maxValue = 0;
    minValue = 0;
}

void GetLogTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    (void)typeSize;
    extraBuf = 0;
    maxLiveNodeCount = 0;
}

void GetLog10MaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    (void)srcShape;
    (void)typeSize;
    (void)isReuseSource;
    maxValue = 0;
    minValue = 0;
}

void GetLog10TmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    (void)typeSize;
    extraBuf = 0;
    maxLiveNodeCount = 0;
}

void GetLog2MaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    (void)isReuseSource;
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    minValue = GetLog2MinTmpSize(typeSize);
    maxValue = GetLog2MaxTmpSize(inputSize, typeSize);
}

void GetLog2TmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    auto calcFactor = (typeSize == sizeof(float)) ? LOG2_FLOAT_CALC_FAC : LOG2_HALF_CALC_FAC;
    extraBuf = 0;
    maxLiveNodeCount = calcFactor;
}
} // namespace AscendC
