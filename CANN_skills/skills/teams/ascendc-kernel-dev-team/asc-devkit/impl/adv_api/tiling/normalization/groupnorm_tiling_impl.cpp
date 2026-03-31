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
 * \file groupnorm_tiling_impl.cpp
 * \brief
 */

#include "include/adv_api/normalization/groupnorm_tiling.h"
#include "../../detail/host_log.h"

namespace optiling {
    REGISTER_TILING_DATA_CLASS(GroupNormTilingOpApi, GroupNormTiling);
} // namespace optiling
namespace AscendC {
namespace {
    constexpr uint32_t GROUPNORM_SRC_DIM_NUM = 4;         // [N, C, H, W]
    constexpr uint32_t GROUPNORM_SIZEOF_FLOAT = 4;
    constexpr uint32_t GROUPNORM_SIZEOF_HALF = 2;
    constexpr uint32_t GROUPNORM_ONE_BLK_SIZE = 32;
    constexpr uint32_t GROUPNORM_THREE_TIMES = 3;
    constexpr uint32_t GROUPNORM_TWO_TIMES = 2;
    constexpr uint32_t GROUPNORM_ONE_NUMBER = 1;
    constexpr uint32_t GROUPNORM_ZERO_NUMBER = 0;
    constexpr float GROUPNORM_ONE_FLOAT_VALUE = 1.0f;

    constexpr uint32_t GROUPNORM_MAX_MASK_VAL = 64;
    constexpr uint32_t GROUPNORM_STEP_MASK_VAL = 8;
    constexpr uint32_t GROUPNORM_MAX_REPEAT_VAL = 255;
    constexpr uint32_t GROUPNORM_MIN_BSCURLENGTH_IN_ITERATION = 8;
    constexpr uint32_t GROUPNORM_REDUCESUM_MAX_FLOAT_NUM = 64;
    constexpr uint32_t GROUPNORM_REDUCESUM_MAX_REPEAT_SMALLSHAPE = 8;

    uint32_t GetGroupNormTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
        uint32_t groupNum, const bool isMaxValue)
    {
        ASCENDC_HOST_ASSERT(typeSize > 0, return 0, "typeSize must be greater than 0.");
        ASCENDC_HOST_ASSERT(groupNum > 0, return 0, "groupNum must be greater than 0.");

        std::vector<int64_t> shapeDims = srcShape.GetDims();
        const uint32_t n = static_cast<uint32_t>(shapeDims[0]);
        const uint32_t c = static_cast<uint32_t>(shapeDims[1]);
        const uint32_t h = static_cast<uint32_t>(shapeDims[2]);
        const uint32_t w = static_cast<uint32_t>(shapeDims[3]);
        ASCENDC_HOST_ASSERT(c != 0, return 0, "the value of c should not be zero!");
        uint32_t mvTmpLen = n * groupNum * sizeof(float);
        uint32_t hwLen = h * w * typeSize;
        uint32_t dhwLen = {0};

        mvTmpLen = (mvTmpLen + GROUPNORM_ONE_BLK_SIZE - GROUPNORM_ONE_NUMBER) / GROUPNORM_ONE_BLK_SIZE *
            GROUPNORM_ONE_BLK_SIZE;
        
        if (isMaxValue) {
            dhwLen = n * c * ((hwLen + GROUPNORM_ONE_BLK_SIZE - GROUPNORM_ONE_NUMBER) / GROUPNORM_ONE_BLK_SIZE *
                GROUPNORM_ONE_BLK_SIZE / typeSize * sizeof(float));
        } else {
            dhwLen = c / groupNum * ((hwLen + GROUPNORM_ONE_BLK_SIZE - GROUPNORM_ONE_NUMBER) / GROUPNORM_ONE_BLK_SIZE *
                GROUPNORM_ONE_BLK_SIZE / typeSize * sizeof(float));
        }

        if (isReuseSource && (typeSize == GROUPNORM_SIZEOF_FLOAT)) {
            return GROUPNORM_TWO_TIMES * dhwLen + GROUPNORM_TWO_TIMES * mvTmpLen;
        }
        return GROUPNORM_THREE_TIMES * dhwLen + GROUPNORM_TWO_TIMES * mvTmpLen;
    }

    void CheckGroupNormHostCommon(const char *apiName, const char *hostFuncName, const ge::Shape& srcShape, 
        const uint32_t typeSize)
    {
        ASCENDC_HOST_ASSERT(srcShape.GetShapeSize() > 0, return, 
            "[%s][%s] Input Shape size must be greater than 0.", apiName, hostFuncName);
        ASCENDC_HOST_ASSERT(srcShape.GetDimNum() == GROUPNORM_SRC_DIM_NUM, return, 
            "[%s][%s] The dims of srcShape is %zu, should be 4 (e.g. [N, C, H, W])!", apiName, hostFuncName, 
            srcShape.GetDimNum());
        ASCENDC_HOST_ASSERT(typeSize == GROUPNORM_SIZEOF_HALF || typeSize == GROUPNORM_SIZEOF_FLOAT, return,
            "[%s][%s] Type size %u is unsupported!", apiName, hostFuncName, typeSize);
        return;
    }
} // namespace name
    
    void GetGroupNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
        const uint32_t groupNum, uint32_t& maxValue, uint32_t& minValue)
    {
        CheckGroupNormHostCommon("GroupNorm", "GetGroupNormMaxMinTmpSize", srcShape, typeSize);
        maxValue = GetGroupNormTmpSize(srcShape, typeSize, isReuseSource, groupNum, true);
        minValue = GetGroupNormTmpSize(srcShape, typeSize, isReuseSource, groupNum, false);
    }

    void GetGroupNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize,
        const bool isReuseSource, const uint32_t groupNum, optiling::GroupNormTiling& tiling) 
    {
        CheckGroupNormHostCommon("GroupNorm", "GetGroupNormNDTilingInfo", srcShape, typeSize);
        ASCENDC_HOST_ASSERT(typeSize > 0, return, "typeSize must be greater than 0.");
        ASCENDC_HOST_ASSERT(groupNum > 0, return, "groupNum must be greater than 0.");

        std::vector<int64_t> shapeDims = srcShape.GetDims();
        const uint32_t n = static_cast<uint32_t>(shapeDims[0]);
        const uint32_t c = static_cast<uint32_t>(shapeDims[1]);
        const uint32_t h = static_cast<uint32_t>(shapeDims[2]);
        const uint32_t w = static_cast<uint32_t>(shapeDims[3]);
        const uint32_t g = groupNum;
        const uint32_t d = c / groupNum;
        ASCENDC_HOST_ASSERT(c != 0, return, "the value of c should not be zero!");
        // HW 32-byte aligned
        const uint32_t hwAlignSize = (typeSize * h * w + GROUPNORM_ONE_BLK_SIZE - GROUPNORM_ONE_NUMBER) / 
            GROUPNORM_ONE_BLK_SIZE * GROUPNORM_ONE_BLK_SIZE / typeSize;
        
        const uint32_t dhwAlignSize = d * hwAlignSize;

        const uint32_t inputXSize = n * c * hwAlignSize;
        const uint32_t meanVarSize = n * g;

        const uint32_t oneBlockNum = GROUPNORM_ONE_BLK_SIZE / GROUPNORM_SIZEOF_FLOAT;
        const uint32_t meanTmpTensorSize = (meanVarSize + oneBlockNum - GROUPNORM_ONE_NUMBER) / oneBlockNum * oneBlockNum;

        uint32_t meanVarTotalSize = 2 * meanTmpTensorSize;
        if (typeSize == GROUPNORM_SIZEOF_FLOAT) {
            meanVarTotalSize = GROUPNORM_ZERO_NUMBER;
        }

        uint32_t numberOfTmpBuf = GROUPNORM_THREE_TIMES;
        if (isReuseSource && (typeSize == GROUPNORM_SIZEOF_FLOAT)) {
            numberOfTmpBuf = GROUPNORM_TWO_TIMES;
        }

        const uint32_t tmpBufSize = stackBufferSize / GROUPNORM_ONE_BLK_SIZE * GROUPNORM_ONE_BLK_SIZE / GROUPNORM_SIZEOF_FLOAT;
        uint32_t oneTmpSize = (tmpBufSize - meanVarTotalSize) / numberOfTmpBuf;

        // The length of a group is one unit.
        uint32_t bsCurLength = oneTmpSize / dhwAlignSize;   

        // determine whether the smallShape calculation is satisfied
        uint32_t k = GROUPNORM_REDUCESUM_MAX_REPEAT_SMALLSHAPE;
        while ((dhwAlignSize / (GROUPNORM_ONE_BLK_SIZE / GROUPNORM_SIZEOF_FLOAT)) % k != 0) {
            k--;
        }
        const bool smallShape = (hwAlignSize <= GROUPNORM_REDUCESUM_MAX_FLOAT_NUM) && 
        (hwAlignSize * d <= GROUPNORM_REDUCESUM_MAX_FLOAT_NUM * k);

        // The constraints imposed by the Level 0 interface of ReduceSum include:
        // Calculating the mask/repeat for two consecutive ReduceSum operations based on DHW.
        // Determining the valid ranges for DHW/bsCurLength.
        if (smallShape) {
            uint32_t mask1{GROUPNORM_MAX_MASK_VAL};
            if (dhwAlignSize > GROUPNORM_MAX_MASK_VAL) {
                while (mask1 != 0 && dhwAlignSize % mask1 != 0) {
                    mask1 -= GROUPNORM_STEP_MASK_VAL;
                }
            } else {
                mask1 = dhwAlignSize;
            }
            ASCENDC_HOST_ASSERT(mask1 > 0, return, "mask1 must be greater than 0.");
            const uint32_t maxBsCurLength = (GROUPNORM_MAX_REPEAT_VAL / (dhwAlignSize / mask1) / 
                GROUPNORM_MIN_BSCURLENGTH_IN_ITERATION) * GROUPNORM_MIN_BSCURLENGTH_IN_ITERATION;
            if (maxBsCurLength < bsCurLength) {
                bsCurLength = maxBsCurLength;
            }
        }

        if (typeSize == GROUPNORM_SIZEOF_HALF && bsCurLength * dhwAlignSize < c) {
            return;
        }

        oneTmpSize = bsCurLength * d * hwAlignSize;

        if (oneTmpSize > inputXSize) {
            bsCurLength = meanVarSize;
            oneTmpSize = inputXSize;
        }

        ASCENDC_HOST_ASSERT((oneTmpSize != GROUPNORM_ZERO_NUMBER), return, "the oneTmpSize should not be zero!");
        if (oneTmpSize == GROUPNORM_ZERO_NUMBER) {
            return;
        }

        const uint32_t inputRoundSize = oneTmpSize;
        const uint32_t inputTailSize = inputXSize % oneTmpSize;

        const uint32_t meanVarRoundSize = inputRoundSize / dhwAlignSize;
        const uint32_t meanVarTailSize = inputTailSize / dhwAlignSize;

        tiling.set_n(n);
        tiling.set_c(c);
        tiling.set_hw(h * w);
        tiling.set_g(g);
        tiling.set_d(d);
        tiling.set_hwAlignSize(hwAlignSize);
        tiling.set_dhwAlignSize(dhwAlignSize);
        tiling.set_inputXSize(inputXSize);
        tiling.set_meanVarSize(meanVarSize);
        tiling.set_numberOfTmpBuf(numberOfTmpBuf);
        tiling.set_meanTmpTensorPos(GROUPNORM_ZERO_NUMBER);
        tiling.set_meanTmpTensorSize(meanTmpTensorSize);
        tiling.set_varianceTmpTensorPos(meanTmpTensorSize);
        tiling.set_varianceTmpTensorSize(meanTmpTensorSize);
        tiling.set_tmpBufSize(tmpBufSize);
        tiling.set_oneTmpSize(oneTmpSize);
        tiling.set_firstTmpStartPos(meanVarTotalSize);
        tiling.set_secondTmpStartPos(meanVarTotalSize + oneTmpSize);
        tiling.set_thirdTmpStartPos(meanVarTotalSize + GROUPNORM_TWO_TIMES * oneTmpSize);
        tiling.set_loopRound(inputXSize / oneTmpSize);
        tiling.set_inputRoundSize(inputRoundSize);
        tiling.set_inputTailSize(inputTailSize);
        tiling.set_inputTailPos(inputXSize - inputTailSize);
        tiling.set_meanVarRoundSize(meanVarRoundSize);
        tiling.set_meanVarTailSize(meanVarTailSize);
        tiling.set_meanVarTailPos(meanVarSize - meanVarTailSize);
        tiling.set_bshCurLength(inputRoundSize);
        tiling.set_bsCurLength(bsCurLength);
        tiling.set_factor(GROUPNORM_ONE_FLOAT_VALUE / (d * h * w));
        tiling.set_smallShape(smallShape);
    }

    void GetGroupNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize,
        const bool isReuseSource, const uint32_t groupNum, AscendC::tiling::GroupNormTiling& tiling) 
    {
        optiling::GroupNormTiling tilingData;
        GetGroupNormNDTilingInfo(srcShape, stackBufferSize, typeSize, isReuseSource, groupNum, tilingData);
        tilingData.SaveToBuffer(&tiling, sizeof(GroupNormTiling));
    }
} // namespace AscendC
