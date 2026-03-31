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
 * \file ascend_quant_tiling_impl.cpp
 * \brief
 */
#include "include/adv_api/quantization/ascend_quant_tiling.h"

#include <cstdint>

#include "graph/tensor.h"
#include "../../detail/host_log.h"
#include "tiling/platform/platform_ascendc.h"
namespace AscendC {
namespace {
constexpr uint32_t ASCEND_QUANT_TWO_TIMES = 2;
constexpr uint32_t ASCEND_QUANT_ONE_REPEAT_BYTE_SIZE = 256;
constexpr uint32_t ASCEND_QUANT_MEMORY_CALC = 2;
constexpr uint32_t QUANT_HALF_SIZE = 2;
constexpr uint32_t QUANT_FLOAT_SIZE = 4;
constexpr uint32_t QUANT_ALIGN_SRC_SIZE = 32;

inline uint32_t GetAscendQuantMaxTmpSize(const uint32_t inputSize)
{
    constexpr uint32_t blkSize = 32;
    uint32_t maxVal = std::max(inputSize * ASCEND_QUANT_MEMORY_CALC,
        ASCEND_QUANT_TWO_TIMES * ASCEND_QUANT_ONE_REPEAT_BYTE_SIZE);
    return (maxVal + blkSize - 1) / blkSize * blkSize;
}

inline uint32_t GetAscendQuantMinTmpSize()
{
    return ASCEND_QUANT_TWO_TIMES * ASCEND_QUANT_ONE_REPEAT_BYTE_SIZE;
}

void CheckQuantHostCommon(const char *apiName, const char *hostFuncName, const ge::Shape& srcShape, 
    const uint32_t typeSize)
{
    ASCENDC_HOST_ASSERT(srcShape.GetShapeSize() > 0, return, 
        "[%s][%s] Input Shape size must be greater than 0.", apiName, hostFuncName);
    ASCENDC_HOST_ASSERT(srcShape.GetShapeSize() * typeSize % QUANT_ALIGN_SRC_SIZE == 0, return, 
        "[%s][%s] Input Shape size must be 32B aligned!.", apiName, hostFuncName);
    ASCENDC_HOST_ASSERT(typeSize == QUANT_HALF_SIZE || typeSize == QUANT_FLOAT_SIZE, return, 
        "[%s][%s] Type size %u is unsupported!", apiName, hostFuncName, typeSize);
    return;
}
} // namespace

void GetAscendQuantTmpBufferFactorSize(uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
{
    extraBuf = 0u;
    maxLiveNodeCount = ASCEND_QUANT_TWO_TIMES;
}

void GetAscendQuantMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue,
    uint32_t& minValue)
{
    CheckQuantHostCommon("AscendQuant", "GetAscendQuantMaxMinTmpSize", srcShape, typeSize);
    maxValue = GetAscendQuantMaxTmpSize(srcShape.GetShapeSize());
    minValue = GetAscendQuantMinTmpSize();
}
} // namespace AscendC
