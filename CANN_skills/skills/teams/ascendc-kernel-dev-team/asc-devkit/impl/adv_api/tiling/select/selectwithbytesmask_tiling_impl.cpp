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
 * \file selectwithbytesmask_tiling_impl.cpp
 * \brief
 */
#include "include/adv_api/select/selectwithbytesmask_tiling.h"
#include "../../detail/host_log.h"

#include <cstdint>
#include <algorithm>

#include "graph/tensor.h"
namespace AscendC {
namespace {
constexpr uint32_t SIZE_OF_HALF = 2;
constexpr uint32_t PER_BLOCK_SIZE = 32;
constexpr uint32_t RESERVED_BUFFER = 512;
constexpr int32_t BASIC_BLOCK_SIZE = 4096;
constexpr uint32_t MIN_TEMP_BUFFER = 1024;
constexpr uint32_t SELECTWITHMASK_DIM_TWO = 2;
constexpr uint32_t SELECTWITHMASK_TYPE_TWO = 2;
constexpr uint32_t SELECTWITHMASK_TYPE_FOUR = 4;

inline uint32_t PaddingTo(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b * b;
}

inline bool IsScalar(const ge::Shape &shape)
{
    return shape.GetDimNum() == 0 || shape.GetShapeSize() == 1;
}

inline uint32_t ComputeBufferSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const ge::Shape& maskShape,
    const uint32_t maskTypeSize, const bool isReuseMask, const bool isMinBuffer)
{
    const int32_t src1Size = src1Shape.GetShapeSize();
    const int32_t src0Size = src0Shape.GetShapeSize();
    const bool isSrc0Scalar = IsScalar(src0Shape);
    const int32_t srcSize = isSrc0Scalar ? src1Size : src0Size;

    const int32_t maskLastAxis = maskShape.GetDim(maskShape.GetDimNum() - 1);
    const int32_t srcLastAxis =
        isSrc0Scalar ? src1Shape.GetDim(src1Shape.GetDimNum() - 1) : src0Shape.GetDim(src0Shape.GetDimNum() - 1);

    const uint32_t calSize = (isMinBuffer && srcSize > BASIC_BLOCK_SIZE) ? BASIC_BLOCK_SIZE : srcSize;
    const uint32_t paddingSrcSize = PaddingTo(calSize, PER_BLOCK_SIZE) * SIZE_OF_HALF;
    uint32_t requiredBuffer = std::max(MIN_TEMP_BUFFER, paddingSrcSize + RESERVED_BUFFER);
    if (maskLastAxis != srcLastAxis && !isReuseMask && maskLastAxis > 1) {
        requiredBuffer += PaddingTo(srcSize * maskTypeSize, PER_BLOCK_SIZE);
    }
    return requiredBuffer;
}

void CheckSelectParams(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize,
    const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask, const char* funcName)
{
    (void)isReuseMask;

    if (!(srcTypeSize == SELECTWITHMASK_TYPE_TWO || srcTypeSize == SELECTWITHMASK_TYPE_FOUR)) {
        TILING_LOG_WARNING("[Select][%s] The srcTypeSize is not involved in the calculation, "
            "its value is %u, should be 2 or 4.", funcName, srcTypeSize);
    }
    ASCENDC_HOST_ASSERT(maskTypeSize == 1 || maskTypeSize == SELECTWITHMASK_TYPE_TWO ||
        maskTypeSize == SELECTWITHMASK_TYPE_FOUR, continue,
        "[Select][%s] The value of maskTypeSize is %u, should be 1 or 2 or 4.", funcName, maskTypeSize);

    bool ans = (src0Shape.GetDimNum() == SELECTWITHMASK_DIM_TWO) ||
        (src1Shape.GetDimNum() == SELECTWITHMASK_DIM_TWO);
    ASCENDC_HOST_ASSERT(ans, continue, "[Select][%s] The src0Shape dimension number is %zu, "
        "the src1Shape dimension number is %zu, one of them should be 2.", funcName,
        src0Shape.GetDimNum(), src1Shape.GetDimNum());
    ASCENDC_HOST_ASSERT(maskShape.GetDimNum() == SELECTWITHMASK_DIM_TWO, continue,
        "[Select][%s] The dims of maskShape is %zu, should be 2.", funcName, maskShape.GetDimNum());
}
} // namespace

uint32_t GetSelectMinTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize,
    const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask)
{
    CheckSelectParams(src0Shape, src1Shape, srcTypeSize, maskShape, maskTypeSize, isReuseMask, "GetSelectMinTmpSize");
    return ComputeBufferSize(src0Shape, src1Shape, maskShape, maskTypeSize, isReuseMask, true);
}

uint32_t GetSelectWithBytesMaskMinTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape,
    const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask)
{
    return GetSelectMinTmpSize(src0Shape, src1Shape, srcTypeSize, maskShape, maskTypeSize, isReuseMask);
}

uint32_t GetSelectMaxTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize,
    const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask)
{
    CheckSelectParams(src0Shape, src1Shape, srcTypeSize, maskShape, maskTypeSize, isReuseMask, "GetSelectMaxTmpSize");
    return ComputeBufferSize(src0Shape, src1Shape, maskShape, maskTypeSize, isReuseMask, false);
}

uint32_t GetSelectWithBytesMaskMaxTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape,
    const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask)
{
    return GetSelectMaxTmpSize(src0Shape, src1Shape, srcTypeSize, maskShape, maskTypeSize, isReuseMask);
}

void GetSelectMaxMinTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize,
    const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask, uint32_t& maxValue,
    uint32_t& minValue)
{
    CheckSelectParams(src0Shape, src1Shape, srcTypeSize, maskShape, maskTypeSize, isReuseMask,
        "GetSelectMaxMinTmpSize");
    maxValue = GetSelectMaxTmpSize(src0Shape, src1Shape, srcTypeSize, maskShape, maskTypeSize, isReuseMask);
    minValue = GetSelectMinTmpSize(src0Shape, src1Shape, srcTypeSize, maskShape, maskTypeSize, isReuseMask);
}

void GetSelectWithBytesMaskMaxMinTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape,
    const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask,
    uint32_t& maxValue, uint32_t& minValue)
{
    GetSelectMaxMinTmpSize(src0Shape, src1Shape, srcTypeSize, maskShape, maskTypeSize, isReuseMask, maxValue, minValue);
}
} // namespace AscendC
