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
 * \file tan_tiling_impl.cpp
 * \brief
 */
#include "include/adv_api/math/tan_tiling.h"
#include <cstdint>

#include "graph/tensor.h"
#include "../../detail/host_log.h"

namespace AscendC {
namespace {
constexpr uint32_t TAN_HALF_CALC_FAC = 10;
constexpr uint32_t TAN_FLOAT_CALC_FAC = 4;
constexpr uint32_t TAN_ONE_REPEAT_BYTE_SIZE = 256;

inline uint32_t GetTanMaxTmpSize(const uint32_t inputSize, const uint32_t typeSize)
{
    const uint32_t calcPro = typeSize == sizeof(float) ? TAN_FLOAT_CALC_FAC : TAN_HALF_CALC_FAC;
    return calcPro * std::max(inputSize * typeSize, TAN_ONE_REPEAT_BYTE_SIZE);
}

inline uint32_t GetTanMinTmpSize(const uint32_t typeSize)
{
    auto typeSizeCalcProc = (typeSize == sizeof(float) ? TAN_FLOAT_CALC_FAC : TAN_HALF_CALC_FAC);
    return TAN_ONE_REPEAT_BYTE_SIZE * typeSizeCalcProc;
}
} // namespace

void GetTanMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    (void)isReuseSource;
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    minValue = GetTanMinTmpSize(typeSize);
    maxValue = GetTanMaxTmpSize(inputSize, typeSize);
}

void GetTanTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    extraBuf = 0;
    maxLiveNodeCount = (typeSize == sizeof(float)) ? TAN_FLOAT_CALC_FAC : TAN_HALF_CALC_FAC;
}
} // namespace AscendC
