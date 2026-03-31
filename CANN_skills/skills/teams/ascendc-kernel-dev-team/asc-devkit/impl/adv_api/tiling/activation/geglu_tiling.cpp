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
 * \file geglu_tiling.cpp
 * \brief
 */
#include "include/adv_api/activation/geglu_tiling.h"

#include <cstdint>
#include <set>

#include "graph/tensor.h"
#include "../../detail/host_log.h"
#include "../../detail/api_check/host_apicheck.h"
namespace AscendC {
namespace {
constexpr uint32_t GEGLU_ONE_REPEAT_BYTE_SIZE = 256;
constexpr uint32_t GEGLU_HALF_CALC_FAC = 4;
constexpr uint32_t GEGLU_FLOAT_CALC_FAC = 0;
static constexpr uint32_t GE_GELU_HALF_SIZE = 2;
static constexpr uint32_t GE_GELU_FLOAT_SIZE = 4;
static const std::set<uint32_t> SUPPORT_TYPESIZE = { GE_GELU_HALF_SIZE, GE_GELU_FLOAT_SIZE };
static constexpr const char GE_GELU_GET_MAX_MIN[] = "GetGeGLUMaxMinTmpSize";
static constexpr const char GE_GELU_GET_TMP_BUFFER[] = "GetGeGLUTmpBufferFactorSize";

inline uint32_t GetGeGLUMaxTmpSize(const uint32_t inputSize, const uint32_t typeSize)
{
    const uint32_t calcPro = typeSize == sizeof(float) ? GEGLU_FLOAT_CALC_FAC : GEGLU_HALF_CALC_FAC;
    return calcPro * std::max(inputSize * typeSize, GEGLU_ONE_REPEAT_BYTE_SIZE);
}

inline uint32_t GetGeGLUMinTmpSize(const uint32_t typeSize)
{
    return typeSize == sizeof(float) ? GEGLU_FLOAT_CALC_FAC * GEGLU_ONE_REPEAT_BYTE_SIZE :
                                       GEGLU_HALF_CALC_FAC * GEGLU_ONE_REPEAT_BYTE_SIZE;
}
} // namespace

void GetGeGLUMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    HighLevelApiCheck::SrcShapeSizeVerifyingParameters<GE_GELU_GET_MAX_MIN>(srcShape.GetShapeSize(), typeSize);
    HighLevelApiCheck::TypeSizeVerifyingParameters<GE_GELU_GET_MAX_MIN>(typeSize, SUPPORT_TYPESIZE);
    HighLevelApiCheck::IsReuseSourceVerifyingParameters<GE_GELU_GET_MAX_MIN>(isReuseSource);
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    minValue = GetGeGLUMinTmpSize(typeSize);
    maxValue = GetGeGLUMaxTmpSize(inputSize, typeSize);
}

void GetGeGLUTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    HighLevelApiCheck::TypeSizeVerifyingParameters<GE_GELU_GET_TMP_BUFFER>(typeSize, SUPPORT_TYPESIZE);
    extraBuf = 0;
    maxLiveNodeCount = (typeSize == sizeof(float)) ? GEGLU_FLOAT_CALC_FAC : GEGLU_HALF_CALC_FAC;
}
} // namespace AscendC
