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
 * \file xor_tiling.cpp
 * \brief
 */
#include "include/adv_api/math/xor_tiling.h"
#include "../../detail/host_log.h"
#include "graph/tensor.h"
#include "register/tilingdata_base.h"
namespace AscendC {
namespace {
constexpr uint32_t XOR_CALC_FAC = 1;
constexpr uint32_t XOR_ONE_REPEAT_BYTE_SIZE = 256;

uint32_t GetXorMaxTmpSize(const ge::Shape srcShape, const uint32_t typeSize, const bool isReuseSource)
{
    (void) isReuseSource;
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    uint32_t inputSize = 1;
    for (const auto dim : shapeDims) {
        inputSize *= dim;
    }
    ASCENDC_HOST_ASSERT(inputSize > 0, return 0, "Input Shape size must be greater than 0.");
    uint8_t calcPro = XOR_CALC_FAC;
    return calcPro * inputSize * typeSize;
}

uint32_t GetXorMinTmpSize(const ge::Shape srcShape, const uint32_t typeSize, const bool isReuseSource)
{
    (void)(srcShape);
    (void)(isReuseSource);
    (void)(typeSize);
    uint8_t calcPro = XOR_CALC_FAC;
    return calcPro * XOR_ONE_REPEAT_BYTE_SIZE;
}
} // namespace

void GetXorMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    uint32_t max = GetXorMaxTmpSize(srcShape, typeSize, isReuseSource);
    minValue = GetXorMinTmpSize(srcShape, typeSize, isReuseSource);
    maxValue = std::max(max, minValue);
}

void GetXorTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuffer)
{
    (void)typeSize;
    auto calcFactor = XOR_CALC_FAC;
    extraBuffer = 0;
    maxLiveNodeCount = calcFactor;
}

}  // namespace AscendC
