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
 * \file reduce_xor_sum_tiling.cpp
 * \brief
 */
#include "include/adv_api/reduce/reduce_xor_sum_tiling.h"
#include "../../detail/host_log.h"

namespace AscendC {
namespace {
constexpr uint32_t REDUCE_XOR_SUM_ONE_REPEAT_BYTE_SIZE = 256;
constexpr uint32_t REDUCE_XOR_SUM_REUSE_CALC_PROC = 2U;
constexpr uint32_t REDUCE_XOR_SUM_NOREUSE_CALC_PROC = 3U;

inline uint32_t GetTmpSize(const uint32_t inputSize, const uint32_t typeSize, const bool isReuseSource)
{
    uint32_t calcPro = isReuseSource ? REDUCE_XOR_SUM_REUSE_CALC_PROC : REDUCE_XOR_SUM_NOREUSE_CALC_PROC;
    return calcPro * std::max(inputSize * typeSize, REDUCE_XOR_SUM_ONE_REPEAT_BYTE_SIZE);
}
}  // namespace

void GetReduceXorSumMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    const uint32_t inputSize = srcShape.GetShapeSize();

    ASCENDC_HOST_ASSERT(inputSize > 0U, return,
        "[ReduceXorSum][GetReduceXorSumMaxMinTmpSize] The parameter srcShape size is %u, expected is greater than 0!",
         inputSize);
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    ASCENDC_HOST_ASSERT(shapeDims.size() > 0UL, return,
        "[ReduceXorSum][GetReduceXorSumMaxMinTmpSize] The parameter srcShape dimension number is %lu, expected is greater than 0!",
         shapeDims.size());
    ASCENDC_HOST_ASSERT(typeSize == 2U, return,
        "[ReduceXorSum][GetReduceXorSumMaxMinTmpSize] The parameter typeSize is %u, expected is 2!", typeSize);

    maxValue = GetTmpSize(inputSize, typeSize, isReuseSource);
    minValue = maxValue;
}
}  // namespace AscendC
