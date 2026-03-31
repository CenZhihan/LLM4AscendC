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
 * \file sum_tiling.cpp
 * \brief
 */
#include "include/adv_api/reduce/sum_tiling.h"
#include "../../detail/host_log.h"

namespace AscendC {
void GetSumMaxMinTmpSize(
    const uint32_t n, const uint32_t typeSize, const bool isReuseSource, uint32_t &maxSize, uint32_t &minSize)
{
    if (isReuseSource) {
        TILING_LOG_WARNING("[Sum][GetSumMaxMinTmpSize] The parameter isReuseSource is true, which is not effective!");
    }
    ASCENDC_HOST_ASSERT(typeSize > 0, return,
        "[Sum][GetSumMaxMinTmpSize] The parameter typeSize is %u, expected is 2 or 4!", typeSize);
    ASCENDC_HOST_ASSERT(n > 0,
         return, "[Sum][GetSumMaxMinTmpSize] The parameter n is %u, expected is greater than 0!", n);

    constexpr uint32_t sumOneBlkSize = 32;
    constexpr uint32_t sumOneRepeatByteSize = 256;

    uint32_t elementNumPerBlk = sumOneBlkSize / typeSize;  // half=16 float=8
    uint32_t elementNumPerRep = sumOneRepeatByteSize / typeSize;

    ASCENDC_HOST_ASSERT((elementNumPerBlk > 0 && elementNumPerRep > 0), return, "typeSize value is too large.");

    uint32_t repeatTimes = (n + elementNumPerRep - 1) / elementNumPerRep;
    uint32_t finalWorkSize = (repeatTimes + elementNumPerBlk - 1) / elementNumPerBlk * elementNumPerBlk * typeSize;
    maxSize = minSize = finalWorkSize;
}
}  // namespace AscendC
