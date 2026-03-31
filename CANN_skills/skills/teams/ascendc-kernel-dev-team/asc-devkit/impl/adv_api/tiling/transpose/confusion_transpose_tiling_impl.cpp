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
 * \file confusion_transpose_tiling_impl.cpp
 * \brief
 */

#include "include/adv_api/transpose/confusion_transpose_tiling.h"
#include "../../detail/host_log.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(ConfusionTransposeTilingOpApi, ConfusionTransposeTiling);
}

namespace AscendC {
enum class TransposeType {
    // default value
    TRANSPOSE_TYPE_NONE,
    // { shape:[B, A1, A3 / 16, A2 / 16, 16, 16], format:"NZ"} -->{ shape:[B, A2, A1, A3], ori_shape:[B, A2, A1, A3],
    // format:"ND"}
    TRANSPOSE_NZ2ND_0213,
    // { shape:[B, A1, A3 / 16, A2 / 16, 16, 16], format:"NZ"}-->{ shape:[B, A2, A3 / 16, A1 / 16, 16, 16],
    // origin_shape:[B, A2, A1, A3], format:"NZ"}
    TRANSPOSE_NZ2NZ_0213,
    // { shape:[B, H / 16, S / 16, 16, 16], format:"NZ"}-->{ shape:[B, N, H/N/16, S / 16, 16, 16], ori_shape:[B, N, S,
    // H/N], format:"NZ"}
    TRANSPOSE_NZ2NZ_012_WITH_N,
    // { shape:[B, H / 16, S / 16, 16, 16], format:"NZ"}-->{ shape:[B, N, S, H/N], ori_shape:[B, N, S, H/N],
    // format:"ND"}
    TRANSPOSE_NZ2ND_012_WITH_N,
    // { shape:[B, N, H/N/16, S/16, 16, 16], format:"NZ"}-->{ shape:[B, S, H], ori_shape:[B, S, H], format:"ND"}
    TRANSPOSE_NZ2ND_012_WITHOUT_N,
    // { shape:[B, N, H/N/16, S/16, 16, 16], format:"NZ"}-->{ shape:[B, H/16, S/16, 16, 16], ori_shape:[B, S, H],
    // format:"NZ"}
    TRANSPOSE_NZ2NZ_012_WITHOUT_N,
    TRANSPOSE_ND2ND_ONLY,    // { shape:[H, W], format:"ND"} -->{ shape:[W, H], format:"ND"}
    TRANSPOSE_ND_UB_GM,      //  [B, N, S, H/N] -> [B, S, H]
    TRANSPOSE_GRAD_ND_UB_GM, //  [B, S, H] -> [B, N, S, H/N]
    TRANSPOSE_ND2ND_B16,     // { shape:[16, 16], format:"ND", dataType: B16} -->{ shape:[16, 16], format:"ND"}
    TRANSPOSE_NCHW2NHWC,     // [ N, C, H, W] -> [N, H, W, C]
    TRANSPOSE_NHWC2NCHW,     // [ N, H, W, C] -> [N, C, H, W]
    TRANSPOSE_ND2ND_021,     // [H, W]->[W, H], [N, H, W]->[N, W, H]
    TRANSPOSE_ND2ND_102,     // [N, H, W]->[H, N, W]
    TRANSPOSE_ND2ND_210,      // [N, H, W]->[W, H, N]
    TRANSPOSE_ND2NZ_WITH_INTLV // [N, D] -> [N, Z]
};

// scene1/2：srcShape[B, A1, A2, A3]
inline void GetConfusionTranspose0213TilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, optiling::ConfusionTransposeTiling& tiling)
{
    ASCENDC_HOST_ASSERT(typeSize > 0, return, "typeSize must be greater than 0.");
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    uint32_t shapeB = shapeDims[0];
    uint32_t shapeA1 = shapeDims[1];
    uint32_t alignA2 = (shapeDims[2] + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE;
    uint32_t widthTiling = (shapeDims[3] + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t alignA3 = widthTiling * BLOCK_CUBE;

    // stackBuffer align to [16,16]
    uint32_t newPopSize = (stackBufferSize * sizeof(uint8_t) / typeSize / CUBE_MAX_SIZE) * CUBE_MAX_SIZE; // element
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t newPopH = newPopSize / BLOCK_CUBE;
    uint32_t needSize = alignA2 * BLOCK_CUBE;
    uint32_t mainBlocks = needSize / newPopSize;
    uint32_t tailSize = needSize % newPopSize;
    uint32_t alignA2MulAlignA3 = alignA2 * alignA3;
    uint32_t batchOffset = shapeA1 * alignA2MulAlignA3;
    uint32_t alignA3MulA1 = shapeA1 * alignA3;
    uint32_t shapeA1BlockCube = shapeA1 * BLOCK_CUBE;
    uint32_t mainOffset = newPopH * mainBlocks * alignA3MulA1;

    tiling.set_param0(blockSize);
    tiling.set_param1(shapeB);
    tiling.set_param2(shapeA1);
    tiling.set_param3(alignA3);
    tiling.set_param4(alignA2);
    tiling.set_param5(widthTiling);
    tiling.set_param6(newPopSize);
    tiling.set_param7(newPopH);
    tiling.set_param8(needSize);
    tiling.set_param9(mainBlocks);
    tiling.set_param10(tailSize);
    tiling.set_param11(alignA2MulAlignA3);
    tiling.set_param12(batchOffset);
    tiling.set_param13(alignA3MulA1);
    tiling.set_param14(shapeA1BlockCube);
    tiling.set_param15(mainOffset);
}

// scene3：srcShape[B, N, S, H/N]
inline void GetConfusionTranspose2NZ012NTilingInfo(const ge::Shape& srcShape, const uint32_t typeSize,
    optiling::ConfusionTransposeTiling& tiling)
{
    ASCENDC_HOST_ASSERT(typeSize > 0, return, "typeSize must be greater than 0.");
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t shapeB = shapeDims[0];
    uint32_t shapeN = shapeDims[1];
    uint32_t hnDiv = shapeDims[3];
    uint32_t blockNum = BLOCK_CUBE / blockSize;
    uint32_t shapeH = hnDiv * shapeN;
    uint32_t hBlockNum = (shapeH + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t sBlockNum = (shapeDims[2] + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t alignH = hBlockNum * BLOCK_CUBE;
    uint32_t alignS = sBlockNum * BLOCK_CUBE;
    uint32_t hnDivBlockNum = (hnDiv + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t alignHnDiv = hnDivBlockNum * BLOCK_CUBE;
    uint32_t gap = alignHnDiv - hnDiv;
    uint32_t alignsBlockCube = alignS * BLOCK_CUBE;
    uint32_t prehBlockNum = shapeN * hnDivBlockNum;
    uint32_t dstBatchOffset = shapeN * alignHnDiv * alignS;
    uint32_t srcBatchOffset = alignH * alignS;

    tiling.set_param0(blockSize);
    tiling.set_param1(shapeB);
    tiling.set_param2(shapeN);
    tiling.set_param3(hnDiv);
    tiling.set_param4(blockNum);
    tiling.set_param5(shapeH);
    tiling.set_param6(hBlockNum);
    tiling.set_param7(sBlockNum);
    tiling.set_param8(alignH);
    tiling.set_param9(alignS);
    tiling.set_param10(hnDivBlockNum);
    tiling.set_param11(alignHnDiv);
    tiling.set_param12(gap);
    tiling.set_param13(alignsBlockCube);
    tiling.set_param14(prehBlockNum);
    tiling.set_param15(dstBatchOffset);
    tiling.set_param16(srcBatchOffset);
}

// scene4：srcShape[B, N, S, H/N]
inline void GetConfusionTranspose2ND012NTilingInfo(const ge::Shape& srcShape, const uint32_t typeSize,
    optiling::ConfusionTransposeTiling& tiling)
{
    ASCENDC_HOST_ASSERT(typeSize > 0, return, "typeSize must be greater than 0.");
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t shapeB = shapeDims[0];
    uint32_t shapeN = shapeDims[1];
    uint32_t hnDiv = shapeDims[3];
    uint32_t shapeH = shapeN * hnDiv;
    uint32_t hBlockNum = (shapeH + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t sBlockNum = (shapeDims[2] + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t hnDivBlockNum = (hnDiv + BLOCK_CUBE - 1) / 16;
    uint32_t alignHnDiv = hnDivBlockNum * BLOCK_CUBE;
    uint32_t gap = alignHnDiv - hnDiv;
    uint32_t alignsCube = sBlockNum * CUBE_MAX_SIZE;
    uint32_t prehBlockNum = shapeN * hnDivBlockNum;
    uint32_t alignsMulAlignHnDiv = sBlockNum * BLOCK_CUBE * alignHnDiv;
    uint32_t alignHnDivCube = alignHnDiv * BLOCK_CUBE;
    uint32_t alignHnDivBlockSize = alignHnDiv * blockSize;
    uint32_t dstBatchOffset = shapeN * alignHnDiv * sBlockNum * BLOCK_CUBE;
    uint32_t srcBatchOffset = hBlockNum * sBlockNum * CUBE_MAX_SIZE;
    uint32_t blockNum = BLOCK_CUBE / blockSize;

    tiling.set_param0(blockSize);
    tiling.set_param1(shapeB);
    tiling.set_param2(shapeN);
    tiling.set_param3(hnDiv);
    tiling.set_param4(shapeH);
    tiling.set_param5(hBlockNum);
    tiling.set_param6(sBlockNum);
    tiling.set_param7(hnDivBlockNum);
    tiling.set_param8(alignHnDiv);
    tiling.set_param9(gap);
    tiling.set_param10(alignsCube);
    tiling.set_param11(prehBlockNum);
    tiling.set_param12(alignsMulAlignHnDiv);
    tiling.set_param13(alignHnDivCube);
    tiling.set_param14(alignHnDivBlockSize);
    tiling.set_param15(dstBatchOffset);
    tiling.set_param16(srcBatchOffset);
    tiling.set_param17(blockNum);
}

// scene5/6：srcShape[B, N, S, H/N]
inline void GetConfusionTranspose012TilingInfo(const ge::Shape& srcShape, const uint32_t typeSize,
    optiling::ConfusionTransposeTiling& tiling)
{
    ASCENDC_HOST_ASSERT(typeSize > 0, return, "typeSize must be greater than 0.");
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t shapeB = shapeDims[0];
    uint32_t shapeN = shapeDims[1];
    uint32_t hnDiv = shapeDims[3];
    uint32_t shapeH = shapeN * hnDiv;
    uint32_t hBlockNum = (shapeH + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t sBlockNum = (shapeDims[2] + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t hnDivBlockNum = (hnDiv + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t alignH = hBlockNum * BLOCK_CUBE;
    uint32_t alignsCube = sBlockNum * CUBE_MAX_SIZE;
    uint32_t alignhBlockCube = alignH * BLOCK_CUBE;
    uint32_t blockSizeMulAlignH = blockSize * alignH;
    uint32_t srcBatchOffset = shapeN * hnDivBlockNum * BLOCK_CUBE * sBlockNum * BLOCK_CUBE;
    uint32_t dstBatchOffset = alignH * sBlockNum * BLOCK_CUBE;
    uint32_t blockNum = BLOCK_CUBE / blockSize;

    tiling.set_param0(blockSize);
    tiling.set_param1(shapeB);
    tiling.set_param2(shapeN);
    tiling.set_param3(hnDiv);
    tiling.set_param4(shapeH);
    tiling.set_param5(hBlockNum);
    tiling.set_param6(sBlockNum);
    tiling.set_param7(hnDivBlockNum);
    tiling.set_param8(alignH);
    tiling.set_param9(alignsCube);
    tiling.set_param10(alignhBlockCube);
    tiling.set_param11(blockSizeMulAlignH);
    tiling.set_param12(srcBatchOffset);
    tiling.set_param13(dstBatchOffset);
    tiling.set_param14(blockNum);
}

// scene7：srcShape[H, W]
void GetConfusionTransposeOnlyTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, optiling::ConfusionTransposeTiling& tiling)
{
    ASCENDC_HOST_ASSERT(typeSize > 0, return, "typeSize must be greater than 0.");
    (void)stackBufferSize;
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t height = shapeDims[0];
    uint32_t width = shapeDims[1];
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.set_param0(blockSize);
    tiling.set_param1(height);
    tiling.set_param2(width);
    tiling.set_param3(highBlock);
    tiling.set_param4(stride);
    tiling.set_param5(repeat);
}

void GetConfusionTransposeOnlyTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, AscendC::tiling::ConfusionTransposeTiling& tiling)
{
    optiling::ConfusionTransposeTiling tilingData;
    GetConfusionTransposeOnlyTilingInfo(srcShape, stackBufferSize, typeSize, tilingData);
    tilingData.SaveToBuffer(&tiling, sizeof(ConfusionTransposeTiling));
}

// scene13
inline void GetConfusionTranspose021TilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, optiling::ConfusionTransposeTiling& tiling)
{
    ASCENDC_HOST_ASSERT(typeSize > 0, return, "typeSize must be greater than 0.");
    (void)stackBufferSize;
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    ASCENDC_HOST_ASSERT((shapeDims.size() == 3) || (shapeDims.size() == 2) , return, "srcShape must be 2 or 3 dims.");
    uint32_t dim0 = 1;
    uint32_t dim1 = shapeDims[0];
    uint32_t dim2 = shapeDims[1];
    constexpr uint32_t max021PatternSize = 3;
    if (shapeDims.size() == max021PatternSize) {
        dim0 = shapeDims[0];
        dim1 = shapeDims[1];
        dim2 = shapeDims[2];
    }

    tiling.set_param0(dim0);
    tiling.set_param1(dim1);
    tiling.set_param2(dim2);
}

// scene14
inline void GetConfusionTranspose102TilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, optiling::ConfusionTransposeTiling& tiling)
{
    ASCENDC_HOST_ASSERT(typeSize > 0, return, "typeSize must be greater than 0.");
    (void)stackBufferSize;
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    ASCENDC_HOST_ASSERT(shapeDims.size() == 3, return, "srcShape must be 3 dims.");
    uint32_t dim0 = shapeDims[0];
    uint32_t dim1 = shapeDims[1];
    uint32_t dim2 = shapeDims[2];

    tiling.set_param0(dim0);
    tiling.set_param1(dim1);
    tiling.set_param2(dim2);
}

// scene15
inline void GetConfusionTranspose210TilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, optiling::ConfusionTransposeTiling& tiling)
{
    ASCENDC_HOST_ASSERT(typeSize > 0, return, "typeSize must be greater than 0.");
    (void)stackBufferSize;
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    ASCENDC_HOST_ASSERT(shapeDims.size() == 3, return, "srcShape must be 3 dims.");
    uint32_t dim0 = shapeDims[0];
    uint32_t dim1 = shapeDims[1];
    uint32_t dim2 = shapeDims[2];

    tiling.set_param0(dim0);
    tiling.set_param1(dim1);
    tiling.set_param2(dim2);
}

inline void GetConfusionTranspose0213MaxMinTmpSize(
    const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
{
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    uint32_t alignA2 = (shapeDims[2] + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE;

    maxValue = alignA2 * BLOCK_CUBE * typeSize;
    minValue = CUBE_MAX_SIZE * typeSize;
}

inline void GetConfusionTranspose2NZ012NMaxMinTmpSize(
    const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
{
    (void)(srcShape);
    maxValue = TWO_TIMES * CUBE_MAX_SIZE * typeSize;
    minValue = TWO_TIMES * CUBE_MAX_SIZE * typeSize;
}

inline void GetConfusionTranspose2ND012NMaxMinTmpSize(
    const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
{
    (void)(srcShape);
    maxValue = TWO_TIMES * CUBE_MAX_SIZE * typeSize;
    minValue = TWO_TIMES * CUBE_MAX_SIZE * typeSize;
}

inline void GetConfusionTranspose012MaxMinTmpSize(
    const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
{
    (void)(srcShape);
    maxValue = TWO_TIMES * CUBE_MAX_SIZE * typeSize;
    minValue = TWO_TIMES * CUBE_MAX_SIZE * typeSize;
}

inline void GetConfusionTransposeOnlyMaxMinTmpSize(
    const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
{
    (void)(srcShape);
    (void)typeSize;
    maxValue = 0;
    minValue = 0;
}

inline void GetConfusionTranspose021MaxMinTmpSize(
    const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
{
    (void)(srcShape);
    (void)typeSize;
    maxValue = 0;
    minValue = 0;
}

inline void GetConfusionTranspose102MaxMinTmpSize(
    const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
{
    (void)(srcShape);
    (void)typeSize;
    maxValue = 0;
    minValue = 0;
}

inline void GetConfusionTranspose210MaxMinTmpSize(
    const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
{
    (void)(srcShape);
    (void)typeSize;
    maxValue = 0;
    minValue = 0;
}

void GetTransposeTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize,
    const uint32_t transposeTypeIn, optiling::ConfusionTransposeTiling& tiling)
{
    if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_NZ2ND_0213 ||
        static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_NZ2NZ_0213) {
        GetConfusionTranspose0213TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    } else if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_NZ2NZ_012_WITH_N) {
        GetConfusionTranspose2NZ012NTilingInfo(srcShape, typeSize, tiling);
    } else if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_NZ2ND_012_WITH_N) {
        GetConfusionTranspose2ND012NTilingInfo(srcShape, typeSize, tiling);
    } else if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_NZ2ND_012_WITHOUT_N ||
        static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N) {
        GetConfusionTranspose012TilingInfo(srcShape, typeSize, tiling);
    } else if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_ND2ND_ONLY) {
        GetConfusionTransposeOnlyTilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    } else if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_ND2ND_021) {
        GetConfusionTranspose021TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    } else if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_ND2ND_102) {
        GetConfusionTranspose102TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    } else if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_ND2ND_210) {
        GetConfusionTranspose210TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    }
}

void GetTransposeTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize,
    const uint32_t transposeTypeIn, AscendC::tiling::ConfusionTransposeTiling& tiling)
{
    optiling::ConfusionTransposeTiling tilingData;
    GetTransposeTilingInfo(srcShape, stackBufferSize, typeSize, transposeTypeIn, tilingData);
    tilingData.SaveToBuffer(&tiling, sizeof(ConfusionTransposeTiling));
}

void GetConfusionTransposeTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize,
    const uint32_t transposeTypeIn, optiling::ConfusionTransposeTiling& tiling)
{
    GetTransposeTilingInfo(srcShape, stackBufferSize, typeSize, transposeTypeIn, tiling);
}

void GetConfusionTransposeTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize,
    const uint32_t transposeTypeIn, AscendC::tiling::ConfusionTransposeTiling& tiling)
{
    GetTransposeTilingInfo(srcShape, stackBufferSize, typeSize, transposeTypeIn, tiling);
}

void GetTransposeMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const uint32_t transposeTypeIn,
    uint32_t& maxValue, uint32_t& minValue)
{
    if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_NZ2ND_0213 ||
        static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_NZ2NZ_0213) {
        GetConfusionTranspose0213MaxMinTmpSize(srcShape, typeSize, maxValue, minValue);
    } else if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_NZ2NZ_012_WITH_N) {
        GetConfusionTranspose2NZ012NMaxMinTmpSize(srcShape, typeSize, maxValue, minValue);
    } else if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_NZ2ND_012_WITH_N) {
        GetConfusionTranspose2ND012NMaxMinTmpSize(srcShape, typeSize, maxValue, minValue);
    } else if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_NZ2ND_012_WITHOUT_N ||
        static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N) {
        GetConfusionTranspose012MaxMinTmpSize(srcShape, typeSize, maxValue, minValue);
    } else if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_ND2ND_ONLY) {
        GetConfusionTransposeOnlyMaxMinTmpSize(srcShape, typeSize, maxValue, minValue);
    } else if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_ND2ND_021) {
        GetConfusionTranspose021MaxMinTmpSize(srcShape, typeSize, maxValue, minValue);
    } else if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_ND2ND_102) {
        GetConfusionTranspose102MaxMinTmpSize(srcShape, typeSize, maxValue, minValue);
    } else if (static_cast<TransposeType>(transposeTypeIn) == TransposeType::TRANSPOSE_ND2ND_210) {
        GetConfusionTranspose210MaxMinTmpSize(srcShape, typeSize, maxValue, minValue);
    }
}

void GetConfusionTransposeMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize,
    const uint32_t transposeTypeIn, uint32_t& maxValue, uint32_t& minValue)
{
    GetTransposeMaxMinTmpSize(srcShape, typeSize, transposeTypeIn, maxValue, minValue);
}

} // namespace AscendC
