/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "include/adv_api/reduce/mean_tiling.h"
#include "register/tilingdata_base.h"
#include "../../detail/host_log.h"

namespace AscendC {
constexpr uint32_t MEAN_CALC_PROC = 1;
const uint32_t MEAN_ONE_BLK_SIZE = 32;
const uint32_t MEAN_ONE_REPEAT_BYTE_SIZE = 256;
const uint32_t HALF_TYPE_SIZE = 2;
const uint32_t FLOAT_TYPE_SIZE = 4;

inline void CheckMeanHostParams(const uint32_t n, const uint32_t srcTypeSize,
    const uint32_t accTypeSize, const bool isReuseSource)
{
    ASCENDC_HOST_ASSERT(
        ((srcTypeSize == 2U && accTypeSize == 2U) ||
         (srcTypeSize == 4U && accTypeSize == 4U) ||
         (srcTypeSize == 2U && accTypeSize == 4U)),
         return,
         "[Mean][GetMeanMaxMinTmpSize] The parameter (srcTypeSize, accTypeSize) is (%u, %u), expected is (2, 2)/(4, 4)/(2, 4).",
         srcTypeSize, accTypeSize
        );
    ASCENDC_HOST_ASSERT(n > 0,
         return, "[Mean][GetMeanMaxMinTmpSize] The parameter n is %u, expected is greater than 0!", n);
    if (isReuseSource) {
        TILING_LOG_WARNING("[Mean][GetMeanMaxMinTmpSize] The parameter isReuseSource is true, which is not effective!");
    }
}

void GetMeanMaxMinTmpSize(const uint32_t n, const uint32_t srcTypeSize,
    const uint32_t accTypeSize, const bool isReuseSource, uint32_t& maxSize, uint32_t& minSize)
{
    CheckMeanHostParams(n, srcTypeSize, accTypeSize, isReuseSource);
    if (srcTypeSize == 0) {
        return;
    }
    uint32_t elementNumPerRep = MEAN_ONE_REPEAT_BYTE_SIZE / srcTypeSize;
    uint32_t repeatTimes = (n + elementNumPerRep - 1) / elementNumPerRep;
    uint32_t finalWorkSize = (repeatTimes + MEAN_ONE_BLK_SIZE - 1) / MEAN_ONE_BLK_SIZE * MEAN_ONE_BLK_SIZE;
    uint32_t inner = (n * srcTypeSize + MEAN_ONE_BLK_SIZE - 1) / MEAN_ONE_BLK_SIZE * MEAN_ONE_BLK_SIZE / srcTypeSize;
    if (srcTypeSize == HALF_TYPE_SIZE && accTypeSize == FLOAT_TYPE_SIZE) {
        elementNumPerRep = MEAN_ONE_REPEAT_BYTE_SIZE / FLOAT_TYPE_SIZE;
        repeatTimes = (n + elementNumPerRep - 1) / elementNumPerRep;
        finalWorkSize = (repeatTimes + MEAN_ONE_BLK_SIZE - 1) / MEAN_ONE_BLK_SIZE * MEAN_ONE_BLK_SIZE +
                        inner * FLOAT_TYPE_SIZE;
    }
    maxSize = minSize = finalWorkSize;
}

void GetMeanTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuffer)
{
    (void)typeSize;
    auto calcFactor = MEAN_CALC_PROC;
    extraBuffer = 0;
    maxLiveNodeCount = calcFactor;
}
}  // namespace AscendC
