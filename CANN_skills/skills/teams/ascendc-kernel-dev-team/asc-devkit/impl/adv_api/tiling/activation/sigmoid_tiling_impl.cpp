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
 * \file sigmoid_tiling_impl.cpp
 * \brief
 */
#include "include/adv_api/activation/sigmoid_tiling.h"

#include <cstdint>
#include <set>

#include "graph/tensor.h"
#include "../../detail/host_log.h"
#include "../../detail/api_check/host_apicheck.h"
namespace AscendC {
namespace {
constexpr uint32_t SIGMOID_MEMORY_SIZE = 256;
static constexpr uint32_t SIGMOID_HALF_SIZE = 2;
static constexpr uint32_t SIGMOID_FLOAT_SIZE = 4;
static const std::set<uint32_t> SUPPORT_TYPESIZE = { SIGMOID_HALF_SIZE, SIGMOID_FLOAT_SIZE };
static constexpr const char SIGMOID_GET_MAX_MIN[] = "GetSigmoidMaxMinTmpSize";
inline uint32_t GetSigmoidMaxTmpSize(const uint32_t inputSize, const uint32_t typeSize)
{
    return inputSize * typeSize > SIGMOID_MEMORY_SIZE ? inputSize * typeSize : SIGMOID_MEMORY_SIZE;
}
} // namespace

void GetSigmoidMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    HighLevelApiCheck::SrcShapeSizeVerifyingParameters<SIGMOID_GET_MAX_MIN>(srcShape.GetShapeSize(), typeSize);
    HighLevelApiCheck::TypeSizeVerifyingParameters<SIGMOID_GET_MAX_MIN>(typeSize, SUPPORT_TYPESIZE);
    HighLevelApiCheck::IsReuseSourceVerifyingParameters<SIGMOID_GET_MAX_MIN>(isReuseSource);
    const uint32_t inputSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");

    maxValue = GetSigmoidMaxTmpSize(inputSize, typeSize);
    minValue = SIGMOID_MEMORY_SIZE;
}
} // namespace AscendC
