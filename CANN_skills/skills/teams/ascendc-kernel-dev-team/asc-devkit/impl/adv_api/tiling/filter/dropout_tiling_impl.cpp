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
 * \file dropout_tiling_impl.h
 * \brief
 */
#include "graph/tensor.h"
#include "register/tilingdata_base.h"
#include "include/adv_api/filter/dropout_tiling.h"
#include "../../detail/host_log.h"

namespace AscendC {
namespace {
constexpr uint32_t DROPOUT_TWO_TIMES = 2;
constexpr uint32_t DROPOUT_ONE_REPEAT_BYTE_SIZE = 256;
constexpr uint32_t DROPOUT_TYPE_TWO = 2;
constexpr uint32_t DROPOUT_TYPE_FOUR = 4;

void CheckDropOutParams(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, const char* funcName)
{
    ASCENDC_HOST_ASSERT(typeSize == DROPOUT_TYPE_TWO || typeSize == DROPOUT_TYPE_FOUR, continue,
        "[DropOut][%s] The value of typeSize is %u, should be 2 or 4.", funcName, typeSize);

    ASCENDC_HOST_ASSERT(srcShape.GetDimNum() > 0, continue,
        "[DropOut][%s] The dims of srcShape is %zu, should be greater than 0.", funcName, srcShape.GetDimNum());

    if (isReuseSource) {
        TILING_LOG_WARNING("[DropOut][%s] The value of isReuseSource is true, may not be effective.", funcName);
    }
}
} // namespace

uint32_t GetDropOutMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource)
{
    CheckDropOutParams(srcShape, typeSize, isReuseSource, "GetDropOutMinTmpSize");

    return DROPOUT_TWO_TIMES * DROPOUT_ONE_REPEAT_BYTE_SIZE;
}

uint32_t GetDropOutMaxTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource)
{
    CheckDropOutParams(srcShape, typeSize, isReuseSource, "GetDropOutMaxTmpSize");

    uint32_t minValue = GetDropOutMinTmpSize(srcShape, typeSize, isReuseSource);

    std::vector<int64_t> shapeDims = srcShape.GetDims();
    uint32_t calculateSize = 1;
    for (uint32_t i = 0; i < shapeDims.size(); i++) {
        calculateSize *= shapeDims[i];
    }

    uint32_t maxValue = DROPOUT_TWO_TIMES * calculateSize * sizeof(float);

    return minValue > maxValue ? minValue : maxValue;
}

void GetDropOutMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    CheckDropOutParams(srcShape, typeSize, isReuseSource, "GetDropOutMaxMinTmpSize");
    maxValue = GetDropOutMaxTmpSize(srcShape, typeSize, isReuseSource);
    minValue = GetDropOutMinTmpSize(srcShape, typeSize, isReuseSource);
}
} // namespace AscendC
