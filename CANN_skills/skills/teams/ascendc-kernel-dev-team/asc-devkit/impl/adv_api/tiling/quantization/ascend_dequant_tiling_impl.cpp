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
 * \file ascend_dequant_tiling_impl.cpp
 * \brief
 */
#include "include/adv_api/quantization/ascend_dequant_tiling.h"

#include <cstdint>

#include "graph/tensor.h"
#include "../../detail/host_log.h"
#include "tiling/platform/platform_ascendc.h"

namespace AscendC {
namespace {

constexpr uint32_t EXPECTED_SHAPE_DIM = 2;      // Expected two dimensional, if 1 dim input, convert to [1, n]
constexpr uint32_t FLOAT_PER_BLOCK = 8;         // 32B  = FP32(4B) * 8
constexpr uint32_t FLOAT_PER_REPEAT = 64;       // 256B = FP32(4B) * 64
constexpr uint32_t DEQUANT_INT32_SIZE = 4;

inline uint32_t GetAscendDequantMaxTmpSize(const uint32_t outer, const uint32_t inner, const uint32_t deqScaleSize)
{
    // need extra 256B to avoid norm mode modify value unexpected or exceeds ub (especially when close to end of ub)
    // Ex: normal mode + Mul with mask[2], might only want to mul 8 FP32, however eventually Mul must read 64 FP32
    //     as a repeat, which might exceeds ub
    uint32_t fp32Num = FLOAT_PER_REPEAT;
    // need FP32 m * n  + FP32 deqScale
    fp32Num += outer * inner + deqScaleSize;
    return fp32Num * sizeof(float);
}

inline uint32_t GetAscendDequantMinTmpSize(const uint32_t outer, const uint32_t inner, const uint32_t deqScaleSize)
{
    (void)outer;
    uint32_t fp32Num = FLOAT_PER_REPEAT;
    // need FP32 1 * n  + FP32 deqScale
    fp32Num += 1 * inner + deqScaleSize;
    return fp32Num * sizeof(float);
}

void CheckDequantHostCommon(const char *apiName, const char *hostFuncName, const ge::Shape& srcShape, 
    const uint32_t typeSize)
{
    ASCENDC_HOST_ASSERT(srcShape.GetShapeSize() > 0, return, 
        "[%s][%s] Input Shape size must be greater than 0.", apiName, hostFuncName);
    ASCENDC_HOST_ASSERT(srcShape.GetDimNum() <= EXPECTED_SHAPE_DIM && srcShape.GetDimNum() > 0, return, 
        "[%s][%s] SrcShape dim %zu is unsupported, must be 1 or 2!", apiName, hostFuncName, srcShape.GetDimNum());
    ASCENDC_HOST_ASSERT(typeSize == DEQUANT_INT32_SIZE, return, 
        "[%s][%s] Type size %u is unsupported!", apiName, hostFuncName, typeSize);
    return;
}
} // namespace

void GetAscendDequantTmpBufferFactorSize(const ge::Shape& srcShape, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    extraBuf = FLOAT_PER_BLOCK + FLOAT_PER_REPEAT;
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    if (shapeDims.size() == EXPECTED_SHAPE_DIM) {
        extraBuf += static_cast<uint32_t>(shapeDims[1]);
        maxLiveNodeCount = 1u;
    } else if (shapeDims.size() == 1) {
        maxLiveNodeCount = EXPECTED_SHAPE_DIM;
    } else {
        extraBuf = 0u;
        maxLiveNodeCount = 0u;
    }
}

void GetAscendDequantMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue,
    uint32_t& minValue)
{
    CheckDequantHostCommon("AscendDequant", "GetAscendDequantMaxMinTmpSize", srcShape, typeSize);
    std::vector<int64_t> shapeDims = srcShape.GetDims();   // should be 2-dimensional or 1-dimensional
    uint32_t outer = 1;
    uint32_t inner = 0;

    if (shapeDims.size() == EXPECTED_SHAPE_DIM) {
        outer = static_cast<uint32_t>(shapeDims[0]);
        inner = static_cast<uint32_t>(shapeDims[1]);
    } else if (shapeDims.size() == 1) {
        inner = static_cast<uint32_t>(shapeDims[0]);
    } else {
        ASCENDC_HOST_ASSERT(false, return, "srcShape should be 1-dimensional / 2-dimensional.");
        return;
    }

    // guarantee deqScaleSize >= deqScale.GetSize()   Ex: uint32_t inner = 8, bf16 deqScale = 16
    const uint32_t deqScaleSize = inner + FLOAT_PER_BLOCK;
    maxValue = GetAscendDequantMaxTmpSize(outer, inner, deqScaleSize);
    minValue = GetAscendDequantMinTmpSize(outer, inner, deqScaleSize);
}
} // namespace AscendC
