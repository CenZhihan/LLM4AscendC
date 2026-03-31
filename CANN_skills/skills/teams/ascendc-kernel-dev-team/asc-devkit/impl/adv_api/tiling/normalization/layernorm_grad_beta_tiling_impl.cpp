/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "include/adv_api/normalization/layernorm_grad_beta_tiling.h"

#include "../../detail/host_log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(LayerNormGradBetaTilingOpApi, LayerNormGradBetaTiling);
} // namespace optiling

namespace AscendC {
namespace {
constexpr uint32_t LAYERNORM_GRAD_BETA_HALF_SIZE = 2;
constexpr uint32_t LAYERNORM_GRAD_BETA_FLOAT_SIZE = 4;
constexpr uint32_t LAYERNORM_GRAD_BETA_SRC_DIM_NUM = 4;

void CheckLayerNormGradBetaHostCommon(const char *apiName, const char *hostFuncName,
    const ge::Shape& srcShape, const uint32_t typeSize)
{
    ASCENDC_HOST_ASSERT(typeSize == LAYERNORM_GRAD_BETA_HALF_SIZE || 
        typeSize == LAYERNORM_GRAD_BETA_FLOAT_SIZE, return,
        "[%s][%s] Type size %u is unsupported!", apiName, hostFuncName, typeSize);
    ASCENDC_HOST_ASSERT(srcShape.GetShapeSize() > 0, return, 
        "[%s][%s] Input Shape size must be greater than 0.", apiName, hostFuncName);
    ASCENDC_HOST_ASSERT(srcShape.GetDimNum() == LAYERNORM_GRAD_BETA_SRC_DIM_NUM, return, 
        "[%s][%s] The dims of srcShape is %zu, should be 4 (e.g. [B, S, storageHLength, originHLength])!", 
        apiName, hostFuncName, srcShape.GetDimNum());
    return;
}

uint32_t GetLayerNormGradBetaMaxTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource)
{
    (void)(isReuseSource);
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    constexpr uint32_t LAYERNORM_GRAD_BETA_SHAPE_SIZE = 4;
    ASCENDC_HOST_ASSERT(shapeDims.size() >= LAYERNORM_GRAD_BETA_SHAPE_SIZE,
        return 0, "srcShape dims must not be less than 4.");
    const uint32_t bLength = static_cast<uint32_t>(shapeDims[LAYERNORM_GRAD_BETA_INDEX_BLENGTH]);
    const uint32_t sLength = static_cast<uint32_t>(shapeDims[LAYERNORM_GRAD_BETA_INDEX_SLENGTH]);
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[LAYERNORM_GRAD_BETA_INDEX_HLENGTH]);

    const uint32_t inputSize = bLength * sLength * hLength * sizeof(float);

    if (typeSize == LAYERNORM_GRAD_BETA_B16_BYTE_SIZE) {
        return LAYERNORM_GRAD_BETA_FOUR_BUF_NUM * inputSize;
    }
    return LAYERNORM_GRAD_BETA_TWO_BUF_NUM * inputSize;
}

uint32_t GetLayerNormGradBetaMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource)
{
    (void)(isReuseSource);
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[LAYERNORM_GRAD_BETA_INDEX_HLENGTH]);

    const uint32_t inputSize = hLength * sizeof(float);

    if (typeSize == LAYERNORM_GRAD_BETA_B16_BYTE_SIZE) {
        return LAYERNORM_GRAD_BETA_FOUR_BUF_NUM * inputSize;
    }
    return LAYERNORM_GRAD_BETA_TWO_BUF_NUM * inputSize;
}

void SetLayerNormGradBetaNDTilingInfo(const LayerNormGradBetaTilingTmp& tilingTmp,
    optiling::LayerNormGradBetaTiling& tiling)
{
    tiling.set_stackBufferSize(tilingTmp.stackBufferSize);
    tiling.set_bLength(tilingTmp.bLength);
    tiling.set_sLength(tilingTmp.sLength);
    tiling.set_hLength(tilingTmp.hLength);
    tiling.set_originalHLength(tilingTmp.originalHLength);
    tiling.set_bshLength(tilingTmp.bshLength);
    tiling.set_bsLength(tilingTmp.bsLength);
    tiling.set_oneCalSize(tilingTmp.oneCalSize);
    tiling.set_numberOfTmpBuf(tilingTmp.numberOfTmpBuf);
    tiling.set_loopRound(tilingTmp.loopRound);
    tiling.set_inputTailSize(tilingTmp.inputTailSize);
    tiling.set_inputTailPos(tilingTmp.inputTailPos);
    tiling.set_bsTailSize(tilingTmp.bsTailSize);
    tiling.set_bshCurLength(tilingTmp.bshCurLength);
    tiling.set_bsCurLength(tilingTmp.bsCurLength);
    tiling.set_gammaTempTensorPos(tilingTmp.gammaTempTensorPos);
    tiling.set_betaTempTensorPos(tilingTmp.betaTempTensorPos);
    tiling.set_inputDyTmpTensorPos(tilingTmp.inputDyTmpTensorPos);
    tiling.set_resForGammaTmpTensorPos(tilingTmp.resForGammaTmpTensorPos);
}
} // namespace

void GetLayerNormGradBetaMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    CheckLayerNormGradBetaHostCommon("LayerNormGradBeta", "GetLayerNormGradBetaMaxMinTmpSize", srcShape, typeSize);
    maxValue = GetLayerNormGradBetaMaxTmpSize(srcShape, typeSize, isReuseSource);
    minValue = GetLayerNormGradBetaMinTmpSize(srcShape, typeSize, isReuseSource);
}

void GetLayerNormGradBetaNDTilingInfo(const ge::Shape srcShape, const uint32_t stackBufferSize, const uint32_t typeSize,
    const bool isReuseSource, optiling::LayerNormGradBetaTiling& tiling)
{
    CheckLayerNormGradBetaHostCommon("LayerNormGradBeta", "GetLayerNormGradBetaNDTilingInfo", srcShape, typeSize);
    LayerNormGradBetaTilingTmp tilingTmp;

    (void)(isReuseSource);
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    constexpr uint32_t LAYERNORM_GRAD_BETA_SHAPE_SIZE = 4;
    ASCENDC_HOST_ASSERT(shapeDims.size() >= LAYERNORM_GRAD_BETA_SHAPE_SIZE,
        return, "srcShape dims must not be less than 4.");
    // detection shapeDims dimension

    tilingTmp.bLength = static_cast<uint32_t>(shapeDims[LAYERNORM_GRAD_BETA_INDEX_BLENGTH]);
    tilingTmp.sLength = static_cast<uint32_t>(shapeDims[LAYERNORM_GRAD_BETA_INDEX_SLENGTH]);
    tilingTmp.hLength = static_cast<uint32_t>(shapeDims[LAYERNORM_GRAD_BETA_INDEX_HLENGTH]);
    tilingTmp.originalHLength = static_cast<uint32_t>(shapeDims[LAYERNORM_GRAD_BETA_INDEX_ORIGINALHLENGTH]);

    tilingTmp.bshLength = tilingTmp.bLength * tilingTmp.sLength * tilingTmp.hLength;
    tilingTmp.bsLength = tilingTmp.bLength * tilingTmp.sLength;

    tilingTmp.stackBufferSize = stackBufferSize / sizeof(float);
    ASCENDC_HOST_ASSERT((tilingTmp.stackBufferSize >= (tilingTmp.hLength + tilingTmp.hLength)),
        return, "stackBufferSize is not enough.");

    tilingTmp.oneCalSize = tilingTmp.stackBufferSize;
    tilingTmp.numberOfTmpBuf = LAYERNORM_GRAD_BETA_ONE_BUF_NUM;

    if (typeSize == LAYERNORM_GRAD_BETA_B16_BYTE_SIZE) {
        tilingTmp.numberOfTmpBuf = LAYERNORM_GRAD_BETA_TWO_BUF_NUM;
        tilingTmp.oneCalSize = tilingTmp.oneCalSize - tilingTmp.hLength - tilingTmp.hLength;
        tilingTmp.oneCalSize = tilingTmp.oneCalSize / tilingTmp.numberOfTmpBuf;
    }
    ASCENDC_HOST_ASSERT(tilingTmp.hLength != 0, return, "the value of hLength should not be zero.");
    tilingTmp.oneCalSize = tilingTmp.oneCalSize / tilingTmp.hLength * tilingTmp.hLength;
    ASCENDC_HOST_ASSERT(tilingTmp.oneCalSize != 0, return, "stackBufferSize is not enough.");

    if (tilingTmp.oneCalSize >= tilingTmp.bshLength) {
        tilingTmp.oneCalSize = tilingTmp.bshLength;
    }

    tilingTmp.loopRound = tilingTmp.bshLength / tilingTmp.oneCalSize;
    tilingTmp.inputTailSize = tilingTmp.bshLength % tilingTmp.oneCalSize;

    tilingTmp.inputTailPos = tilingTmp.bshLength - tilingTmp.inputTailSize;
    tilingTmp.bsTailSize = tilingTmp.inputTailSize / tilingTmp.hLength;

    tilingTmp.bshCurLength = tilingTmp.oneCalSize;
    tilingTmp.bsCurLength = tilingTmp.oneCalSize / tilingTmp.hLength;

    tilingTmp.gammaTempTensorPos = 0;
    tilingTmp.betaTempTensorPos = 0;
    tilingTmp.inputDyTmpTensorPos = 0;
    tilingTmp.resForGammaTmpTensorPos = 0;

    if (typeSize == LAYERNORM_GRAD_BETA_B16_BYTE_SIZE) {
        tilingTmp.gammaTempTensorPos = 0;
        tilingTmp.betaTempTensorPos = tilingTmp.gammaTempTensorPos + tilingTmp.hLength;
        tilingTmp.inputDyTmpTensorPos = tilingTmp.betaTempTensorPos + tilingTmp.hLength;
        tilingTmp.resForGammaTmpTensorPos = tilingTmp.inputDyTmpTensorPos + tilingTmp.oneCalSize;
    }

    SetLayerNormGradBetaNDTilingInfo(tilingTmp, tiling);
}

void GetLayerNormGradBetaNDTilingInfo(const ge::Shape srcShape, const uint32_t stackBufferSize, const uint32_t typeSize,
    const bool isReuseSource, AscendC::tiling::LayerNormGradBetaTiling& tiling)
{
    optiling::LayerNormGradBetaTiling tilingData;
    GetLayerNormGradBetaNDTilingInfo(srcShape, stackBufferSize, typeSize, isReuseSource, tilingData);
    tilingData.SaveToBuffer(&tiling, sizeof(LayerNormGradBetaTiling));
}
} // namespace AscendC
