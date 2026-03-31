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
 * \file dequantize_tiling_impl.cpp
 * \brief
 */
#include "include/adv_api/quantization/dequantize_tiling.h"

#include <cstdint>

#include "graph/tensor.h"
#include "../../detail/host_log.h"

namespace AscendC {
namespace {

constexpr uint32_t EXPECTED_SHAPE_DIM = 2;      // Expected two dimensional, if 1 dim input, convert to [1, n]
constexpr uint32_t FLOAT_PER_BLOCK = 8;         // 32B  = FP32(4B) * 8
constexpr uint32_t FLOAT_PER_REPEAT = 64;       // 256B = FP32(4B) * 64

inline uint32_t GetDequantizeMaxTmpSize(const uint32_t outer, const uint32_t inner, const uint32_t scaleSize)
{
    // need extra 256B to avoid norm mode modify value unexpected or exceeds ub (especially when close to end of ub)
    // Ex: normal mode + Mul with mask[2], might only want to mul 8 FP32, however eventually Mul must read 64 FP32
    //     as a repeat, which might exceeds ub
    uint32_t fp32Num = FLOAT_PER_REPEAT;
    // need FP32 m * n  + FP32 scale
    fp32Num += outer * inner + scaleSize;
    return fp32Num * static_cast<uint32_t>(sizeof(float));
}

inline uint32_t GetDequantizeMinTmpSize(const uint32_t outer, const uint32_t inner, const uint32_t scaleSize)
{
    (void)outer;
    uint32_t fp32Num = FLOAT_PER_REPEAT;
    // need FP32 1 * n  + FP32 scale
    fp32Num += 1U * inner + scaleSize;
    return fp32Num * static_cast<uint32_t>(sizeof(float));
}
} // namespace

void GetDequantizeTmpBufferFactorSize(const ge::Shape& srcShape, uint32_t& maxLiveNodeCnt, uint32_t& extraBuf)
{
    extraBuf = FLOAT_PER_BLOCK + FLOAT_PER_REPEAT;
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    if (shapeDims.size() == EXPECTED_SHAPE_DIM) {
        extraBuf += static_cast<uint32_t>(shapeDims[1]);
        maxLiveNodeCnt = 1u;
    } else if (shapeDims.size() == 1) {
        maxLiveNodeCnt = EXPECTED_SHAPE_DIM;
    } else {
        extraBuf = 0u;
        maxLiveNodeCnt = 0u;
    }
}

void GetDequantizeMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue,
    uint32_t& minValue)
{
    (void)typeSize;
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

    // guarantee scaleSize >= scale.GetSize()   Ex: uint32_t inner = 8, bf16 scale = 16
    const uint32_t scaleSize = inner + FLOAT_PER_BLOCK;
    maxValue = GetDequantizeMaxTmpSize(outer, inner, scaleSize);
    minValue = GetDequantizeMinTmpSize(outer, inner, scaleSize);
}
} // namespace AscendC
