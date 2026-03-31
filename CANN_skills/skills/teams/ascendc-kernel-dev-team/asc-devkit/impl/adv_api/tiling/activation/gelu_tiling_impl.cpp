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
 * \file gelu_tiling_impl.cpp
 * \brief
 */

#include "include/adv_api/activation/gelu_tiling.h"

#include <set>

#include "graph/tensor.h"
#include "register/tilingdata_base.h"
#include "../../detail/api_check/host_apicheck.h"

namespace AscendC {
constexpr uint32_t GELU_THREE_TIMES = 3;
constexpr uint32_t GELU_ONE_REPEAT_BYTE_SIZE = 256;
static constexpr uint32_t GELU_HALF_SIZE = 2;
static constexpr uint32_t GELU_FLOAT_SIZE = 4;
static const std::set<uint32_t> SUPPORT_TYPESIZE = { GELU_HALF_SIZE, GELU_FLOAT_SIZE };
static constexpr const char GELU_GET_MAX[] = "GetGeluMaxTmpSize";
static constexpr const char GELU_GET_MIN[] = "GetGeluMinTmpSize";
static constexpr const char GELU_GET_MAX_MIN[] = "GetGeluMaxMinTmpSize";

uint32_t GetGeluMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize)
{
    HighLevelApiCheck::SrcShapeSizeVerifyingParameters<GELU_GET_MIN>(srcShape.GetShapeSize(), typeSize);
    HighLevelApiCheck::TypeSizeVerifyingParameters<GELU_GET_MIN>(typeSize, SUPPORT_TYPESIZE);
    return GELU_THREE_TIMES * GELU_ONE_REPEAT_BYTE_SIZE;
}

uint32_t GetGeluMaxTmpSize(const ge::Shape& srcShape, const uint32_t typeSize)
{
    HighLevelApiCheck::SrcShapeSizeVerifyingParameters<GELU_GET_MAX>(srcShape.GetShapeSize(), typeSize);
    HighLevelApiCheck::TypeSizeVerifyingParameters<GELU_GET_MAX>(typeSize, SUPPORT_TYPESIZE);
    uint32_t minValue = GetGeluMinTmpSize(srcShape, typeSize);
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    uint32_t calculateSize = 1;
    for (uint32_t i = 0; i < shapeDims.size(); i++) {
        calculateSize *= shapeDims[i];
    }

    uint32_t maxValue = GELU_THREE_TIMES * calculateSize * sizeof(float);

    return minValue > maxValue ? minValue : maxValue;
}

void GetGeluMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue,
    uint32_t& minValue)
{
    HighLevelApiCheck::SrcShapeSizeVerifyingParameters<GELU_GET_MAX_MIN>(srcShape.GetShapeSize(), typeSize);
    HighLevelApiCheck::TypeSizeVerifyingParameters<GELU_GET_MAX_MIN>(typeSize, SUPPORT_TYPESIZE);
    maxValue = GetGeluMaxTmpSize(srcShape, typeSize);
    minValue = GetGeluMinTmpSize(srcShape, typeSize);
}
} // namespace AscendC
