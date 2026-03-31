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
 * \file clamp_tiling_impl.cpp
 * \brief
 */
#include "include/adv_api/math/clamp_tiling.h"
#include <cstdint>
#include "graph/tensor.h"
#include "../../detail/host_log.h"
namespace AscendC {
constexpr uint32_t CLAMP_CALC_FAC = 1;
constexpr uint32_t CLAMP_FLOAT_ELE = 64;
constexpr uint32_t CLAMP_HALF_ELE = 128;

void GetClampMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    (void)isReuseSource;
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    if (typeSize == sizeof(float)) {
        minValue = CLAMP_FLOAT_ELE * sizeof(uint8_t);
        maxValue = std::max<uint64_t>(inputSize * sizeof(uint8_t), CLAMP_FLOAT_ELE * sizeof(uint8_t));
    } else {
        minValue = CLAMP_HALF_ELE * sizeof(uint8_t);
        maxValue = std::max<uint64_t>(inputSize * sizeof(uint8_t), CLAMP_HALF_ELE * sizeof(uint8_t));
    }
}

void GetClampTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    (void)typeSize;
    extraBuf = 0;
    maxLiveNodeCount = CLAMP_CALC_FAC;
}
} // namespace AscendC
