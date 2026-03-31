/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "include/adv_api/math/round_tiling.h"

#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../detail/host_log.h"
namespace AscendC {
constexpr uint32_t ROUND_HALF_CALC_FAC_200 = 2;
constexpr uint32_t ROUND_HALF_CALC_FAC_220 = 1;
constexpr uint32_t ROUND_FLOAT_CALC_FAC = 1;
constexpr uint32_t ROUND_ONE_REPEAT_BYTE_SIZE = 256;

void GetRoundMaxMinTmpSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const ge::Shape& srcShape,
    const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
{
    (void)isReuseSource;
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    const auto npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201) {
        if (typeSize == sizeof(float)) {
            minValue = 0;
            maxValue = 0;
        } else {
            minValue = ROUND_ONE_REPEAT_BYTE_SIZE * ROUND_HALF_CALC_FAC_220;
            maxValue = ROUND_HALF_CALC_FAC_220 * std::max(inputSize * typeSize, ROUND_ONE_REPEAT_BYTE_SIZE);
        }
    } else if (npuArch == NpuArch::DAV_2002) {
        if (typeSize == sizeof(float)) {
            minValue = ROUND_ONE_REPEAT_BYTE_SIZE * ROUND_FLOAT_CALC_FAC;
            maxValue = ROUND_FLOAT_CALC_FAC * std::max(inputSize * typeSize, ROUND_ONE_REPEAT_BYTE_SIZE);
        } else {
            minValue = ROUND_ONE_REPEAT_BYTE_SIZE * ROUND_HALF_CALC_FAC_200;
            maxValue = ROUND_HALF_CALC_FAC_200 * std::max(inputSize * typeSize, ROUND_ONE_REPEAT_BYTE_SIZE);
        }
    }
}

void GetRoundTmpBufferFactorSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const uint32_t typeSize,
    uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    extraBuf = 0;
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    constexpr uint32_t liveNodeTwo = 2;
    if (npuArch == NpuArch::DAV_2201) {
        if (typeSize == sizeof(float)) {
            maxLiveNodeCount = 0;
        } else {
            maxLiveNodeCount = 1;
        }
    } else if (npuArch == NpuArch::DAV_2002) {
        if (typeSize == sizeof(float)) {
            maxLiveNodeCount = 1;
        } else {
            maxLiveNodeCount = liveNodeTwo;
        }
    }
}
} // namespace AscendC
