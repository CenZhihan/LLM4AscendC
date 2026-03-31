/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "../../detail/host_log.h"
#include "include/adv_api/normalization/deepnorm_tiling.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(DeepNormTilingOpApi, DeepNormTiling);
} // namespace optiling
namespace AscendC {
namespace {
constexpr uint32_t ONE_BLK_SIZE = 32;
constexpr uint32_t ONE_BLOCK_NUM = ONE_BLK_SIZE / sizeof(float);  // 1 block = 32 bytes, contains 8 FP32
constexpr uint32_t DEEPNORM_THREE_TIMES = 3;                      // normal constant 3
constexpr uint32_t DEEPNORM_TWO_TIMES = 2;                        // normal constant 2
constexpr uint32_t DEEPNORM_MAX_REPEAT = 255;                     // uint8_t range [0, 255]
constexpr float DEEPNORM_LAST_DIM_INIT_VALUE = 1.0;               // reciprocal base

constexpr uint32_t BASIC_BLK_HLENGTH = 64;            // basic block H must be 64
constexpr uint32_t BASIC_BLK_BSLENGTH = 8;            // basic block B*S must be divisible by 64
constexpr uint32_t DEEPNORM_HALF_SIZE = 2;
constexpr uint32_t DEEPNORM_FLOAT_SIZE = 4;
constexpr uint32_t DEEPNORM_SRC_DIM_NUM = 3;            // [B, S, H]
constexpr uint32_t DEEPNORM_ALIGN_LAST_DIM = 32;        // H align 32B

// ceil(dataAmount / blockSize) * blockSize      blocknum * blockSize needed to fill in dataAmount
uint32_t GetFinalBlockSize(uint32_t dataAmount, uint32_t blockSize)
{
    return (dataAmount + blockSize - 1) / blockSize * blockSize;
}

uint32_t GetDeepNormMaxTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource)
{
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    const uint32_t bLength = static_cast<uint32_t>(shapeDims[0]);
    const uint32_t sLength = static_cast<uint32_t>(shapeDims[1]);
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[2]);

    // applied tensors are all FP32
    uint32_t inputBSsize = bLength * sLength * sizeof(float);
    uint32_t inputBSHsize = bLength * sLength * hLength * sizeof(float);
    inputBSsize = GetFinalBlockSize(inputBSsize, ONE_BLK_SIZE);
    inputBSHsize = GetFinalBlockSize(inputBSHsize, ONE_BLK_SIZE);

    // copy one temp BSH tensor to output Tensor
    if (isReuseSource && (typeSize == sizeof(float))) {
        return DEEPNORM_TWO_TIMES * inputBSHsize + DEEPNORM_TWO_TIMES * inputBSsize;
    }
    return DEEPNORM_THREE_TIMES * inputBSHsize + DEEPNORM_TWO_TIMES * inputBSsize;
}

uint32_t GetDeepNormMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    const bool isBasicBlock)
{
    (void)isBasicBlock;
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    const uint32_t bLength = static_cast<uint32_t>(shapeDims[0]);
    const uint32_t sLength = static_cast<uint32_t>(shapeDims[1]);
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[2]);

    // applied tensors are all FP32
    uint32_t inputBSsize = bLength * sLength;
    uint32_t inputHsize = hLength;
    inputBSsize = GetFinalBlockSize(inputBSsize * sizeof(float), ONE_BLK_SIZE);
    inputHsize = GetFinalBlockSize(inputHsize * sizeof(float), ONE_BLK_SIZE);

    if (isReuseSource && (typeSize == sizeof(float))) {
        return DEEPNORM_TWO_TIMES * inputHsize + DEEPNORM_TWO_TIMES * inputBSsize;
    }
    return DEEPNORM_THREE_TIMES * inputHsize + DEEPNORM_TWO_TIMES * inputBSsize;
}

void CheckDeepNormHostCommon(const char *apiName, const char *hostFuncName, const ge::Shape& srcShape, 
    const uint32_t typeSize, const bool isBasicBlock)
{
    ASCENDC_HOST_ASSERT(srcShape.GetShapeSize() > 0, return, 
        "[%s][%s] Input Shape size must be greater than 0.", apiName, hostFuncName);
    ASCENDC_HOST_ASSERT(typeSize == DEEPNORM_HALF_SIZE || typeSize == DEEPNORM_FLOAT_SIZE, return,
        "[%s][%s] Type size %u is unsupported!", apiName, hostFuncName, typeSize);
    ASCENDC_HOST_ASSERT(srcShape.GetDimNum() == DEEPNORM_SRC_DIM_NUM, return, 
        "[%s][%s] The dims of srcShape is %zu, should be 3 (e.g. [B, S, H])!", apiName, hostFuncName, 
        srcShape.GetDimNum());
    uint32_t bsLength = static_cast<uint32_t>(srcShape.GetDim(0) * srcShape.GetDim(1));
    uint32_t hLength = static_cast<uint32_t>(srcShape.GetDim(2));
    ASCENDC_HOST_ASSERT(hLength > 0, return, 
        "[%s][%s] The last axis H must be greater than 0!", apiName, hostFuncName);
    ASCENDC_HOST_ASSERT(bsLength > 0, return, 
        "[%s][%s] The non-last axis B*S must be greater than 0!", apiName, hostFuncName);
    ASCENDC_HOST_ASSERT(hLength * typeSize % DEEPNORM_ALIGN_LAST_DIM == 0, return,
        "[%s][%s] The last axis H must be 32B aligned!", apiName, hostFuncName);
    ASCENDC_HOST_ASSERT(hLength <= 2040, return,
        "[%s][%s] The value of last axis can not be greater than 2040!", apiName, hostFuncName);
    if (isBasicBlock) {
        ASCENDC_HOST_ASSERT(hLength % BASIC_BLK_HLENGTH == 0, return,
            "[%s][%s] The last axis H must be 64 aligned!", apiName, hostFuncName);
        ASCENDC_HOST_ASSERT(bsLength % BASIC_BLK_BSLENGTH == 0, return,
            "[%s][%s] The non-last axis B*S must be 8 aligned!", apiName, hostFuncName);
    }
    return;
}

void CheckDeepNormTilingInfo(const char *apiName, const char *hostFuncName, const ge::Shape& srcShape, 
    const ge::Shape& originSrcShape, const bool isBasicBlock)
{
    uint32_t hLength = static_cast<uint32_t>(srcShape.GetDim(2));
    uint32_t hOriginLength = static_cast<uint32_t>(originSrcShape.GetDim(2));
    ASCENDC_HOST_ASSERT(hOriginLength > 0 && hOriginLength <= hLength, return, 
        "[%s][%s] The range of originH is %u, must be (0, %u]!", apiName, hostFuncName, hOriginLength, hLength);
    if (isBasicBlock) {
        ASCENDC_HOST_ASSERT(hOriginLength == hLength, return, 
            "[%s][%s] When isBasicBlock is true, originH is %u, H is %u, originH must be equal to H!", 
            apiName, hostFuncName, hOriginLength, hLength);
    }
    return;
}
} // namespace

bool GetDeepNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    const bool isBasicBlock, uint32_t& maxValue, uint32_t& minValue)
{
    CheckDeepNormHostCommon("DeepNorm", "GetDeepNormMaxMinTmpSize", srcShape, typeSize, isBasicBlock);
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    const uint32_t bLength = static_cast<uint32_t>(shapeDims[0]);
    const uint32_t sLength = static_cast<uint32_t>(shapeDims[1]);
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[2]);

    // if basic block case   check   1. hlength must be divisible by 64   2. B*S must be divisible by 8
    bool hDivBy64 = (hLength % BASIC_BLK_HLENGTH == 0) && (hLength > 0);
    bool bsDivBy8 = (bLength * sLength) % BASIC_BLK_BSLENGTH == 0;
    if (isBasicBlock && !(hDivBy64 && bsDivBy8)) {
        return false;
    }

    maxValue = GetDeepNormMaxTmpSize(srcShape, typeSize, isReuseSource);
    minValue = GetDeepNormMinTmpSize(srcShape, typeSize, isReuseSource, isBasicBlock);
    return true;
}

bool GetDeepNormTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, const bool isReuseSource, const bool isBasicBlock, optiling::DeepNormTiling& tiling)
{
    CheckDeepNormHostCommon("DeepNorm", "GetDeepNormTilingInfo", srcShape, typeSize, isBasicBlock);
    CheckDeepNormTilingInfo("DeepNorm", "GetDeepNormTilingInfo", srcShape, originSrcShape, isBasicBlock);
    uint32_t maxSize = 0;
    uint32_t minSize = 0;
    // allocated buffer must be at least minSize
    const bool res = GetDeepNormMaxMinTmpSize(srcShape, typeSize, isReuseSource, isBasicBlock, maxSize, minSize);
    if (!res || stackBufferSize < minSize) {
        return false;
    }

    std::vector<int64_t> shapeDims = srcShape.GetDims();
    const uint32_t bLength = static_cast<uint32_t>(shapeDims[0]);
    const uint32_t sLength = static_cast<uint32_t>(shapeDims[1]);
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[2]);            // H after alignment

    std::vector<int64_t> shapeDimsOri = originSrcShape.GetDims();
    const uint32_t bLengthOri = static_cast<uint32_t>(shapeDimsOri[0]);
    const uint32_t sLengthOri = static_cast<uint32_t>(shapeDimsOri[1]);
    const uint32_t originalHLength = static_cast<uint32_t>(shapeDimsOri[2]); // H before alignment

    bool bsCheck = (bLength == bLengthOri) && (sLength == sLengthOri);       // check originb = b, origins = s
    bool hAlign = (hLength * typeSize % ONE_BLK_SIZE) == 0;                  // Hlength must align with 32 bytes
    bool hCheckOri = (0u < originalHLength  && originalHLength <= hLength);  // oriHLength must be in range (0, Hlength]
    bool basicblkOriH = !(isBasicBlock && originalHLength != hLength);       // when basicblock, it must be oriH = H
    bool hLengthCheck = (hLength <= DEEPNORM_MAX_REPEAT * 8);      // one addimpl has uint8_t repeatStride = hLength / 8

    if (!(bsCheck && hAlign && hCheckOri && basicblkOriH && hLengthCheck)) {
        return false;
    }

    const uint32_t inputXSize = bLength * sLength * hLength;
    const uint32_t meanVarSize = bLength * sLength;

    // if reuse, one tmp buffer can be replaced by output tensor
    uint32_t numberOfTmpBuf = (isReuseSource && (typeSize == sizeof(float)))? DEEPNORM_TWO_TIMES: DEEPNORM_THREE_TIMES;

    // mean and variance are both B * S
    constexpr uint32_t meanTmpTensorPos = 0;
    const uint32_t meanTmpTensorSize = GetFinalBlockSize(meanVarSize, ONE_BLOCK_NUM);
    const uint32_t varianceTmpTensorPos = meanTmpTensorSize;
    const uint32_t varianceTmpTensorSize = meanTmpTensorSize;

    uint32_t meanVarTotalSize = (typeSize == sizeof(float))? 0 : meanTmpTensorSize + varianceTmpTensorSize;
    const uint32_t tmpBufSize = stackBufferSize / sizeof(float);    // how many FP32 tmpBuffer can store in total
    uint32_t oneTmpSize = (tmpBufSize - meanVarTotalSize) / numberOfTmpBuf;  // FP32 num for each H tensor tmpTensorABC
    // basicBlock also means tmpBuffer must be n * hLength
    oneTmpSize = oneTmpSize / hLength * hLength;                             // floor(hLength)
    oneTmpSize = (oneTmpSize > inputXSize) ? inputXSize : oneTmpSize;
    // assume that tmpSize is 9*n, for basicblock, use 8*n is better
    if (isBasicBlock && (oneTmpSize / (BASIC_BLK_BSLENGTH * hLength) > 0)) {
        oneTmpSize = oneTmpSize / (BASIC_BLK_BSLENGTH * hLength) * (BASIC_BLK_BSLENGTH * hLength);
    }

    // stackBufferSize cannot allocate enough space for tmpTensor A, B, C
    if (oneTmpSize == 0) {
        return false;
    }

    const uint32_t firstTmpStartPos = meanVarTotalSize;                  // tmpTensorA
    const uint32_t secondTmpStartPos = firstTmpStartPos + oneTmpSize;    // tmpTensorB
    const uint32_t thirdTmpStartPos = secondTmpStartPos + oneTmpSize;    // tmpTensorC

    const uint32_t loopRound = inputXSize / oneTmpSize;
    const uint32_t inputTailSize = inputXSize % oneTmpSize;
    const uint32_t inputTailPos = inputXSize - inputTailSize;
    const uint32_t inputRoundSize = oneTmpSize;                      // B * S * H every round
    const uint32_t meanVarRoundSize = inputRoundSize / hLength;      // B * S every round
    const uint32_t meanVarTailSize = inputTailSize / hLength;
    const uint32_t meanVarTailPos = meanVarSize - meanVarTailSize;

    const uint32_t bshCurLength = inputRoundSize;
    const uint32_t bsCurLength = meanVarRoundSize;

    const float lastDimValueBack = DEEPNORM_LAST_DIM_INIT_VALUE / static_cast<float>(originalHLength);

    tiling.set_bLength(bLength);
    tiling.set_sLength(sLength);
    tiling.set_hLength(hLength);
    tiling.set_originalHLength(originalHLength);
    tiling.set_inputXSize(inputXSize);
    tiling.set_meanVarSize(meanVarSize);
    tiling.set_numberOfTmpBuf(numberOfTmpBuf);
    tiling.set_meanTmpTensorPos(meanTmpTensorPos);
    tiling.set_meanTmpTensorSize(meanTmpTensorSize);
    tiling.set_varianceTmpTensorPos(varianceTmpTensorPos);
    tiling.set_varianceTmpTensorSize(varianceTmpTensorSize);
    tiling.set_tmpBufSize(tmpBufSize);
    tiling.set_oneTmpSize(oneTmpSize);
    tiling.set_firstTmpStartPos(firstTmpStartPos);
    tiling.set_secondTmpStartPos(secondTmpStartPos);
    tiling.set_thirdTmpStartPos(thirdTmpStartPos);
    tiling.set_loopRound(loopRound);
    tiling.set_inputRoundSize(inputRoundSize);
    tiling.set_inputTailSize(inputTailSize);
    tiling.set_inputTailPos(inputTailPos);
    tiling.set_meanVarRoundSize(meanVarRoundSize);
    tiling.set_meanVarTailSize(meanVarTailSize);
    tiling.set_meanVarTailPos(meanVarTailPos);
    tiling.set_bshCurLength(bshCurLength);
    tiling.set_bsCurLength(bsCurLength);
    tiling.set_lastDimValueBack(lastDimValueBack);
    return true;
}

bool GetDeepNormTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, const bool isReuseSource, const bool isBasicBlock, AscendC::tiling::DeepNormTiling& tiling)
{
    optiling::DeepNormTiling tilingData;
    bool ret = GetDeepNormTilingInfo(srcShape, originSrcShape, stackBufferSize, typeSize, isReuseSource, isBasicBlock, tilingData);
    tilingData.SaveToBuffer(&tiling, sizeof(DeepNormTiling));
    return ret;
}
} // namespace AscendC
