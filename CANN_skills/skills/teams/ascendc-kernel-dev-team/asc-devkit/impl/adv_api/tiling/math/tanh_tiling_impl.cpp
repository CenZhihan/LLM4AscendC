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
 * \file tanh_tiling_impl.cpp
 * \brief
 */
#include <cstdint>
#include "../../detail/host_log.h"
#include "graph/tensor.h"
#include "include/adv_api/math/tanh_tiling.h"

namespace AscendC {
namespace {
constexpr uint32_t TANH_HALF_CALC_PROC = 4;
constexpr uint32_t TANH_FLOAT_CALC_PROC = 1;
constexpr uint32_t TANH_ONE_REPEAT_BYTE_SIZE = 256;

inline uint32_t GetTanhMaxTmpSize(const uint32_t inputSize, const uint32_t typeSize)
{
    const uint8_t calcPro = typeSize == sizeof(float) ? TANH_FLOAT_CALC_PROC : TANH_HALF_CALC_PROC;
    return inputSize * typeSize > TANH_ONE_REPEAT_BYTE_SIZE ?
        calcPro * inputSize * typeSize :
        calcPro * TANH_ONE_REPEAT_BYTE_SIZE; // All temporary variables are float.
}

inline uint32_t GetTanhMinTmpSize(const uint32_t typeSize)
{
    return TANH_ONE_REPEAT_BYTE_SIZE * (typeSize == sizeof(float) ? TANH_FLOAT_CALC_PROC : TANH_HALF_CALC_PROC);
}
} // namespace

void GetTanhMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    (void)isReuseSource;
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    minValue = GetTanhMinTmpSize(typeSize);
    maxValue = GetTanhMaxTmpSize(inputSize, typeSize);
}

void GetTanhTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    extraBuf = 0;
    maxLiveNodeCount = (typeSize == sizeof(float)) ? TANH_FLOAT_CALC_PROC : TANH_HALF_CALC_PROC;
}
} // namespace AscendC
