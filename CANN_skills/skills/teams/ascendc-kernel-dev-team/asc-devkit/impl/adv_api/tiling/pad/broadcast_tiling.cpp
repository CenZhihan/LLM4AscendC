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

#include "include/adv_api/pad/broadcast_tiling.h"
#include "../../detail/host_log.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
namespace AscendC {
namespace {
constexpr uint32_t ONE_BLK_SIZE = 32;
constexpr uint32_t BRCB_ONE_SIZE = 8;
constexpr uint32_t HALF_ONE_BLK_SIZE = 16;
constexpr uint32_t CAST_DIM_TWO = 2;
constexpr uint32_t CAST_TYPE_TWO = 2;
constexpr uint32_t CAST_TYPE_FOUR = 4;

uint32_t GetCastTempBuffer(const ge::Shape& srcShape, const ge::Shape& dstShape, uint32_t& typeSize)
{
    uint32_t castTempBuffer = 0;
    if (typeSize == sizeof(int8_t)) {
        typeSize = sizeof(int16_t);
        const uint32_t srcSize = srcShape.GetShapeSize();
        const uint32_t dstSize = dstShape.GetShapeSize();
        const uint32_t alignSrcSize = ((srcSize + HALF_ONE_BLK_SIZE - 1) / HALF_ONE_BLK_SIZE) * HALF_ONE_BLK_SIZE;
        const uint32_t alignDstSize = ((dstSize + HALF_ONE_BLK_SIZE - 1) / HALF_ONE_BLK_SIZE) * HALF_ONE_BLK_SIZE;
        castTempBuffer = (alignSrcSize + alignDstSize) * typeSize;
    }
    return castTempBuffer;
}

void GetBroadCastMaxMinTmpSize220(
    const ge::Shape& srcShape, const ge::Shape& dstShape, uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
{
    uint32_t castTempBuffer = GetCastTempBuffer(srcShape, dstShape, typeSize);
    const uint32_t oneBlockElementNum = ONE_BLK_SIZE / typeSize;
    bool isAligned = true;
    const uint32_t firstDim = dstShape.GetDim(0);
    const uint32_t numBlocks = dstShape.GetDim(1);
    if (numBlocks % oneBlockElementNum != 0) {
        isAligned = false;
    }
    if (isAligned) {
        const uint32_t minBrcbTempBufferSize = oneBlockElementNum * oneBlockElementNum * typeSize;
        minValue = minBrcbTempBufferSize + castTempBuffer;

        const uint32_t firstDimAlignNum = (firstDim + BRCB_ONE_SIZE - 1) / BRCB_ONE_SIZE * BRCB_ONE_SIZE;
        const uint32_t maxBrcbTempBufferSize = firstDimAlignNum * oneBlockElementNum * typeSize;
        maxValue = maxBrcbTempBufferSize + castTempBuffer;
    } else {
        const uint32_t dstRepeatSize = (numBlocks + oneBlockElementNum - 1) / oneBlockElementNum;
        const uint32_t numBlocksAlign = dstRepeatSize * oneBlockElementNum;
        const uint32_t minBrcbTempBufferSize = oneBlockElementNum * oneBlockElementNum * typeSize;
        const uint32_t minCopyTempBufferSize = oneBlockElementNum * numBlocksAlign * typeSize;
        minValue = minBrcbTempBufferSize + minCopyTempBufferSize + castTempBuffer;

        const uint32_t firstDimAlignNum = (firstDim + BRCB_ONE_SIZE - 1) / BRCB_ONE_SIZE * BRCB_ONE_SIZE;
        const uint32_t maxBrcbTempBufferSize = firstDimAlignNum * oneBlockElementNum * typeSize;
        const uint32_t maxCopyTempBufferSize = firstDim * numBlocksAlign * typeSize;
        maxValue = maxBrcbTempBufferSize + maxCopyTempBufferSize + castTempBuffer;
    }
}

void CheckBroadCastParams(const platform_ascendc::PlatformAscendC& ascendcPlatform, const ge::Shape& srcShape,
    const ge::Shape& dstShape, uint32_t typeSize, const bool isReuseSource, const char* funcName)
{
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_2002) {
        ASCENDC_HOST_ASSERT(typeSize == 1 || typeSize == CAST_TYPE_TWO || typeSize == CAST_TYPE_FOUR, continue,
            "[Broadcast][%s] The value of typeSize is %u, should be 1 or 2 or 4.", funcName, typeSize);

        ASCENDC_HOST_ASSERT(srcShape.GetDimNum() == 1 || srcShape.GetDimNum() == CAST_DIM_TWO, continue,
            "[Broadcast][%s] The dims of srcShape is %zu, should be 1 or 2.", funcName, srcShape.GetDimNum());

        ASCENDC_HOST_ASSERT(srcShape.GetDimNum() == dstShape.GetDimNum(), continue,
            "[Broadcast][%s] The dims of srcShape is %zu, the dims of dstShape is %zu, they should be equal.", funcName,
            srcShape.GetDimNum(), dstShape.GetDimNum());

        if (isReuseSource) {
            TILING_LOG_WARNING("[Broadcast][%s] The value of isReuseSource is true, may not be effective.", funcName);
        }
    }
}
} // namespace

void GetBroadCastMaxMinTmpSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const ge::Shape& srcShape,
    const ge::Shape& dstShape, uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
{
    CheckBroadCastParams(ascendcPlatform, srcShape, dstShape, typeSize, isReuseSource, "GetBroadCastMaxMinTmpSize");

    size_t dstShapeDimNum = dstShape.GetDimNum();
    constexpr uint32_t needDstShapeDimNum = 2;
    auto npuArch = ascendcPlatform.GetCurNpuArch();
        if (npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_2002) {
        ASCENDC_HOST_ASSERT((dstShapeDimNum == needDstShapeDimNum || dstShapeDimNum == 1),
            return, "Now only support dim = 1 or 2.");
    }
    ASCENDC_HOST_ASSERT((srcShape.GetDimNum() == dstShapeDimNum),
        return, "SrcShape dim num and dstShape dim num should be same.");
    ASCENDC_HOST_ASSERT(typeSize > 0, return, "TypeSize must be greater than 0.");
    for (size_t i = 0; i < dstShapeDimNum; i++) {
        ASCENDC_HOST_ASSERT((srcShape.GetDim(i) == dstShape.GetDim(i) || srcShape.GetDim(i) == 1),
            return, "SrcShape cannot broadcast to dstShape.");
    }

    auto srcSize = srcShape.GetShapeSize();
    ASCENDC_HOST_ASSERT(srcSize > 0, return, "SrcSize must be greater than 0.");

    if (srcSize == dstShape.GetShapeSize() || srcSize == 1) {
        if (typeSize != 1) {
            minValue = 0;
            maxValue = 0;
            return;
        }
        const uint32_t castTempBuffer = GetCastTempBuffer(srcShape, dstShape, typeSize);
        minValue = castTempBuffer + ONE_BLK_SIZE;
        maxValue = castTempBuffer + ONE_BLK_SIZE;
        return;
    }

    int32_t axis = 0;
    if (srcShape.GetDim(1) == 1) {
        axis = 1;
    }

    if (npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_2002) {
        if (axis == 0) {
            const uint32_t castTempBuffer = GetCastTempBuffer(srcShape, dstShape, typeSize);
            ASCENDC_HOST_ASSERT(dstShape.GetDim(1) % (ONE_BLK_SIZE / typeSize) == 0,
                return, "dim[1] is not 32-byte aligned.");
            minValue = ONE_BLK_SIZE + castTempBuffer;
            maxValue = ONE_BLK_SIZE + castTempBuffer;
            return;
        }
    }

    if (npuArch == NpuArch::DAV_2201) {
        GetBroadCastMaxMinTmpSize220(srcShape, dstShape, typeSize, maxValue, minValue);
    } else if (npuArch == NpuArch::DAV_2002) {
        const uint32_t castTempBuffer = GetCastTempBuffer(srcShape, dstShape, typeSize);
        const uint32_t oneBlockElementNum = ONE_BLK_SIZE / typeSize;
        ASCENDC_HOST_ASSERT(dstShape.GetDim(0) % oneBlockElementNum == 0,
            return, "dim[0] is not 32-byte aligned.");
        const uint32_t minTmpBufferSize =
            oneBlockElementNum * ((dstShape.GetDim(1) + BRCB_ONE_SIZE - 1) / BRCB_ONE_SIZE) * typeSize;
        minValue = ONE_BLK_SIZE + castTempBuffer + minTmpBufferSize;
        const uint32_t maxTmpufferSize = dstShape.GetDim(0) * dstShape.GetDim(1) * typeSize;
        maxValue = ONE_BLK_SIZE + castTempBuffer + maxTmpufferSize;
    }
}
}  // namespace AscendC
