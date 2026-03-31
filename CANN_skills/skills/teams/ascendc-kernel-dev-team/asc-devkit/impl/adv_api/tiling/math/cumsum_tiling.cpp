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
 * \file broadcast_tiling.cpp
 * \brief
 */

#include "include/adv_api/math/cumsum_tiling.h"
#include "graph/tensor.h"
#include "../../detail/host_log.h"
namespace AscendC {
void GetCumSumMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isLastAxis,
    const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
{
    (void)isReuseSource;
    size_t srcShapeDimNum = srcShape.GetDimNum();
    ASCENDC_HOST_ASSERT((srcShapeDimNum == 2), return, "Now only support dim = 2.");  // 2 is for 2 dim
    ASCENDC_HOST_ASSERT((typeSize == sizeof(uint16_t) || typeSize == sizeof(float)),
        return, "Now only support half and float.");
    constexpr uint32_t oneBlockSize = 32;
    ASCENDC_HOST_ASSERT(((srcShape.GetDim(1) > 0) && ((srcShape.GetDim(1) * typeSize) % oneBlockSize == 0)),
        return, "The inner dim should be 32B aligned.");

    if (!isLastAxis) {
        if (typeSize == sizeof(uint16_t)) {
            maxValue = minValue = srcShape.GetDim(0) * srcShape.GetDim(1) * sizeof(float);
        } else {
            maxValue = 0;
            minValue = 0;
        }
    } else {
        constexpr uint32_t transDataTo5HDAddrListSize = 16;
        uint32_t minCastTempBuffer = 0;
        uint32_t maxCastTempBuffer = 0;
        uint32_t alignOutter = (srcShape.GetDim(0) + transDataTo5HDAddrListSize - 1) / transDataTo5HDAddrListSize *
                               transDataTo5HDAddrListSize;
        constexpr uint32_t transposeTimes = 2;
        if (typeSize == sizeof(uint16_t)) {
            // Only add needs to be converted to float, and the tempBuffer of add can be reused with the tempBuffer
            // of the second transpose, so multiply by sizeof(half).
            minCastTempBuffer = srcShape.GetDim(1) * transDataTo5HDAddrListSize * sizeof(uint16_t);
            maxCastTempBuffer = srcShape.GetDim(1) * alignOutter * sizeof(uint16_t);
        }
        // transpose does not need to convert half to float
        minValue = minCastTempBuffer + transDataTo5HDAddrListSize * srcShape.GetDim(1) * typeSize *
                   transposeTimes;  // Both transpose require tempBuffer

        maxValue = maxCastTempBuffer + alignOutter * srcShape.GetDim(1) * typeSize *
                   transposeTimes;  // Both transpose require tempBuffer
    }
    return;
}
}  // namespace AscendC
