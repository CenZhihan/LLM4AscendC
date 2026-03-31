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
 * \file quantize_tiling_impl.cpp
 * \brief
 */

#include <cstdint>

#include "graph/tensor.h"
#include "../../detail/host_log.h"
#include "include/adv_api/quantization/quantize_tiling.h"
namespace AscendC {
namespace {
constexpr uint32_t QUANTIZE_TWO_TIMES = 2u;
constexpr uint32_t QUANTIZE_ONE_REPEAT_BYTE_SIZE = 256u;
constexpr uint32_t QUANTIZE_MEMORY_CALC = 2u;

inline uint32_t GetQuantizeMaxTmpSize(const uint32_t inputSize)
{
    constexpr uint32_t blkSize = 32;
    uint32_t maxVal = std::max(inputSize * QUANTIZE_MEMORY_CALC,
        QUANTIZE_TWO_TIMES * QUANTIZE_ONE_REPEAT_BYTE_SIZE);
    return (maxVal + blkSize - 1) / blkSize * blkSize;
}

inline uint32_t GetQuantizeMinTmpSize()
{
    return QUANTIZE_TWO_TIMES * QUANTIZE_ONE_REPEAT_BYTE_SIZE;
}
} // namespace

void GetQuantizeTmpBufferFactorSize(uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    extraBuf = 0u;
    maxLiveNodeCount = QUANTIZE_TWO_TIMES;
}

void GetQuantizeMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue,
    uint32_t& minValue)
{
    (void)typeSize;
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    maxValue = GetQuantizeMaxTmpSize(inputSize);
    minValue = GetQuantizeMinTmpSize();
}
} // namespace AscendC
