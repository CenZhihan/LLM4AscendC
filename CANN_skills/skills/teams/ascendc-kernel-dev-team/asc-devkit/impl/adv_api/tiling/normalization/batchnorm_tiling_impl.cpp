/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "include/adv_api/normalization/batchnorm_tiling.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(BatchNormTilingOpApi, BatchNormTiling);
} // namespace optiling
namespace AscendC {
namespace {
constexpr uint32_t BATCHNORM_SIZEOF_FLOAT = 4;
constexpr uint32_t BATCHNORM_SIZEOF_HALF = 2;
constexpr uint32_t FLOAT_BLOCK_NUMBER = 8;
constexpr uint32_t BATCHNORM_HALF_ELE = 16;
constexpr uint32_t BATCHNORM_THREE_TIMES = 3;
constexpr uint32_t BATCHNORM_TWO_TIMES = 2;
constexpr uint32_t BATCHNORM_ONE_BLK_SIZE = 32;
constexpr uint32_t BATCHNORM_ZERO_NUMBER = 0;
constexpr float BATCHNORM_LAST_DIM_INIT_VALUE = 1.0;
constexpr uint32_t BASIC_FLOAT_BLK_SHLENGTH = 64;
constexpr uint32_t MAX_REPEAT_TIMES = 255;
const uint8_t DEFAULT_REPEAT_STRIDE = 8;
constexpr uint32_t BATCHNORM_TWO_BLK_SIZE = 64;
constexpr uint32_t B_INDEX = 0;
constexpr uint32_t S_INDEX = 1;
constexpr uint32_t H_INDEX = 2;
constexpr uint32_t SHAPE_DIM = 3;

inline uint32_t AlignToBlockSize(const uint32_t inputX, const uint32_t inputY)
{
    return (inputX + inputY - 1) / inputY * inputY;
}

inline bool CheckBasicBlockShape(const uint32_t originalBLength, const uint32_t sLength, const uint32_t hLength)
{
    if ((sLength * hLength % BASIC_FLOAT_BLK_SHLENGTH != 0) || (originalBLength % FLOAT_BLOCK_NUMBER != 0)) {
        return false;
    }
    return true;
}

bool CheckShape(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t typeSize,
    const bool isBasicBlock = false)
{
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    if (shapeDims.size() != SHAPE_DIM) {
        return false;
    }
    const uint32_t sLength = static_cast<uint32_t>(shapeDims[S_INDEX]);
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[H_INDEX]);

    std::vector<int64_t> originDims = originSrcShape.GetDims();
    if (originDims.size() != SHAPE_DIM) {
        return false;
    }
    const uint32_t originalBLength = static_cast<uint32_t>(originDims[B_INDEX]);
    // sLength * hLength should align to block-size
    if (sLength * hLength * typeSize % BATCHNORM_ONE_BLK_SIZE != 0) {
        return false;
    }
    // for basic block shlength should be multiples of 64, and originalBLength should be multiples of 8
    if (isBasicBlock && (!CheckBasicBlockShape(originalBLength, sLength, hLength))) {
        return false;
    }
    return true;
}

uint32_t GetBatchNormMaxTmpSize(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t typeSize,
    const bool isReuseSource, const bool isBasicBlock)
{
    (void)typeSize;
    (void)isReuseSource;
    (void)isBasicBlock;
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    std::vector<int64_t> oriShapeDims = originSrcShape.GetDims();
    const uint32_t originalBLength = static_cast<uint32_t>(oriShapeDims[B_INDEX]);
    const uint32_t sLength = static_cast<uint32_t>(shapeDims[S_INDEX]);
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[H_INDEX]);

    uint32_t mvTmpLen = sLength * hLength * BATCHNORM_SIZEOF_FLOAT;
    uint32_t inputLen = originalBLength * sLength * hLength * BATCHNORM_SIZEOF_FLOAT;

    mvTmpLen = AlignToBlockSize(mvTmpLen, BATCHNORM_ONE_BLK_SIZE);
    inputLen = AlignToBlockSize(inputLen, BATCHNORM_ONE_BLK_SIZE);

    return BATCHNORM_THREE_TIMES * inputLen + BATCHNORM_TWO_TIMES * mvTmpLen;
}

uint32_t GetBatchNormMinTmpSize(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t typeSize,
    const bool isReuseSource, const bool isBasicBlock)
{
    (void)isReuseSource;
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    std::vector<int64_t> oriShapeDims = originSrcShape.GetDims();
    const uint32_t originalBLength = static_cast<uint32_t>(oriShapeDims[B_INDEX]);
    const uint32_t sLength = static_cast<uint32_t>(shapeDims[S_INDEX]);
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[H_INDEX]);

    uint32_t mvTmpLen = sLength * hLength * BATCHNORM_SIZEOF_FLOAT;
    uint32_t bBlockLengthDiv = originalBLength * BATCHNORM_ONE_BLK_SIZE;
    uint32_t minBasicBlock = originalBLength * BASIC_FLOAT_BLK_SHLENGTH * BATCHNORM_SIZEOF_FLOAT;

    mvTmpLen = AlignToBlockSize(mvTmpLen, BATCHNORM_ONE_BLK_SIZE);

    if (typeSize == sizeof(uint16_t)) {
        bBlockLengthDiv = originalBLength * BATCHNORM_TWO_BLK_SIZE;
    }

    if (isBasicBlock) {
        bBlockLengthDiv = minBasicBlock;
    }

    return BATCHNORM_THREE_TIMES * bBlockLengthDiv + BATCHNORM_TWO_TIMES * mvTmpLen;
}
} // namespace

bool GetBatchNormMaxMinTmpSize(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t typeSize,
    const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue, const bool isBasicBlock)
{
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    std::vector<int64_t> oriShapeDims = originSrcShape.GetDims();
    const uint32_t originalBLength = static_cast<uint32_t>(oriShapeDims[B_INDEX]);
    const uint32_t sLength = static_cast<uint32_t>(shapeDims[S_INDEX]);
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[H_INDEX]);
    // for basic block shlength should be multiples of 64, and originalBLength should be multiples of 8
    if (isBasicBlock && (!CheckBasicBlockShape(originalBLength, sLength, hLength))) {
        return false;
    }
    maxValue = GetBatchNormMaxTmpSize(srcShape, originSrcShape, typeSize, isReuseSource, isBasicBlock);
    minValue = GetBatchNormMinTmpSize(srcShape, originSrcShape, typeSize, isReuseSource, isBasicBlock);
    return true;
}

bool GetBatchNormNDTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape,
    const uint32_t stackBufferByteSize, const uint32_t typeSize, const bool isReuseSource,
    optiling::BatchNormTiling& tilling, const bool isBasicBlock)
{
    constexpr uint32_t halfBlockNumber = 16;
    if (!CheckShape(srcShape, originSrcShape, typeSize, isBasicBlock)) {
        return false;
    }
    uint32_t minSize = 0;
    uint32_t maxSize = 0;
    bool res =
        GetBatchNormMaxMinTmpSize(srcShape, originSrcShape, typeSize, isReuseSource, maxSize, minSize, isBasicBlock);
    if (!res || stackBufferByteSize < minSize) {
        return false;
    }
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    std::vector<int64_t> oriShapeDims = originSrcShape.GetDims();
    const uint32_t sLength = static_cast<uint32_t>(shapeDims[1]);
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[2]);
    const uint32_t originalBLength = static_cast<uint32_t>(oriShapeDims[0]);
    const uint32_t originalInputXSize = originalBLength * sLength * hLength;
    const uint32_t meanVarSize = sLength * hLength;
    uint32_t numberOfTmpBuf = BATCHNORM_THREE_TIMES;
    constexpr uint32_t meanTmpTensorPos = BATCHNORM_ZERO_NUMBER;
    const uint32_t meanTmpTensorSize = AlignToBlockSize(meanVarSize, FLOAT_BLOCK_NUMBER);
    const uint32_t varianceTmpTensorPos = meanTmpTensorSize;
    const uint32_t varianceTmpTensorSize = meanTmpTensorSize;
    uint32_t meanVarTotalSize = meanTmpTensorSize + varianceTmpTensorSize;
    if (typeSize == BATCHNORM_SIZEOF_FLOAT) {
        meanVarTotalSize = BATCHNORM_ZERO_NUMBER;
    }
    const uint32_t tmpBufSize = stackBufferByteSize / BATCHNORM_SIZEOF_FLOAT;
    uint32_t oneTmpSize = (tmpBufSize - meanVarTotalSize) / numberOfTmpBuf;
    if (typeSize != BATCHNORM_SIZEOF_FLOAT) {
        oneTmpSize = oneTmpSize / (originalBLength * BATCHNORM_HALF_ELE) * (originalBLength * BATCHNORM_HALF_ELE);
    } else {
        oneTmpSize = oneTmpSize / (originalBLength * FLOAT_BLOCK_NUMBER) * (originalBLength * FLOAT_BLOCK_NUMBER);
    }
    if (oneTmpSize > originalInputXSize) {
        oneTmpSize = originalInputXSize;
    }
    if (oneTmpSize == BATCHNORM_ZERO_NUMBER) {
        return false;
    }
    uint32_t shCurLength = oneTmpSize / originalBLength;
    const uint32_t shCurLengthAlign = shCurLength / BASIC_FLOAT_BLK_SHLENGTH * BASIC_FLOAT_BLK_SHLENGTH;
    if (isBasicBlock && (shCurLength % BASIC_FLOAT_BLK_SHLENGTH != 0)) {
        shCurLength = shCurLengthAlign;
        oneTmpSize = shCurLength * originalBLength;
    }
    const uint32_t firstTmpStartPos = meanVarTotalSize;
    const uint32_t secondTmpStartPos = firstTmpStartPos + oneTmpSize;
    const uint32_t thirdTmpStartPos = secondTmpStartPos + oneTmpSize;
    const uint32_t loopRound = originalInputXSize / oneTmpSize;
    const uint32_t inputTailSize = originalInputXSize % oneTmpSize;
    const uint32_t inputTailPos = (originalInputXSize - inputTailSize) / originalBLength;
    const uint32_t meanVarTailSize = inputTailSize / originalBLength;
    const uint32_t meanVarTailPos = meanVarSize - meanVarTailSize;
    const uint32_t bshCurLength = oneTmpSize;
    float firstDimValueBack = BATCHNORM_LAST_DIM_INIT_VALUE / static_cast<float>(originalBLength);
    const uint32_t castHalfRepStride = DEFAULT_REPEAT_STRIDE / BATCHNORM_SIZEOF_HALF;
    const uint32_t shCurLengthBlockNum = shCurLength / FLOAT_BLOCK_NUMBER;
    const uint32_t castHalfOutRepStride = meanVarSize / halfBlockNumber;
    tilling.set_originalBLength(originalBLength);
    tilling.set_meanVarSize(meanVarSize);
    tilling.set_meanTmpTensorPos(meanTmpTensorPos);
    tilling.set_varianceTmpTensorPos(varianceTmpTensorPos);
    tilling.set_tmpBufSize(tmpBufSize);
    tilling.set_oneTmpSize(oneTmpSize);
    tilling.set_firstTmpStartPos(firstTmpStartPos);
    tilling.set_secondTmpStartPos(secondTmpStartPos);
    tilling.set_thirdTmpStartPos(thirdTmpStartPos);
    tilling.set_loopRound(loopRound);
    tilling.set_inputTailSize(inputTailSize);
    tilling.set_inputTailPos(inputTailPos);
    tilling.set_meanVarTailSize(meanVarTailSize);
    tilling.set_meanVarTailPos(meanVarTailPos);
    tilling.set_bshCurLength(bshCurLength);
    tilling.set_shCurLength(shCurLength);
    tilling.set_firstDimValueBack(firstDimValueBack);
    tilling.set_castHalfRepStride(castHalfRepStride);
    tilling.set_shCurLengthBlockNum(shCurLengthBlockNum);
    tilling.set_castHalfOutRepStride(castHalfOutRepStride);
    return true;
}

bool GetBatchNormNDTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape,
    const uint32_t stackBufferByteSize, const uint32_t typeSize, const bool isReuseSource,
    AscendC::tiling::BatchNormTiling& tilling, const bool isBasicBlock)
{
    optiling::BatchNormTiling tillingData;
    bool ret = GetBatchNormNDTilingInfo(srcShape, originSrcShape, stackBufferByteSize, typeSize, isReuseSource,
                                    tillingData, isBasicBlock);
    tillingData.SaveToBuffer(&tilling, sizeof(BatchNormTiling));
    return ret;
}
} // namespace AscendC
