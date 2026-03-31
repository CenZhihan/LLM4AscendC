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
 * \file conv3d_tiling_util.cpp
 * \brief
 */

#include "conv3d_tiling_util.h"

namespace Conv3dTilingApi {

int64_t LCM(int64_t numL, int64_t numR) {
    if (numR == 0 || numL == 0) {
        return 1;
    }
    int64_t product = numL * numR;
    while (numL % numR != 0) {
        int64_t tmp = numL % numR;
        numL = numR;
        numR = tmp;
    }

    return product / numR;
}

uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return b;
    }
    return (a + b - 1) / b;
}

uint64_t AlignB(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

uint64_t Gcd(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }

    uint64_t c;
    if (a < b) {
        c = a;
        a = b;
        b = c;
    }

    while (a % b != 0) {
        c = a % b;
        a = b;
        b = c;
    }

    return b;
}

void CalcCommFactorWithPowerOfTwo(const uint64_t num, const uint64_t numMax, std::vector<uint64_t> &resList)
{
    uint64_t sqrtMax = static_cast<uint64_t>(sqrt(num));
    for (uint64_t i = 1; i <= sqrtMax; ++i) {
        if (num % i == 0) {
            if (i <= numMax) {
                resList.emplace_back(i);
            }
            uint64_t right = num / i;
            if (right != i && right <= numMax) {
                resList.emplace_back(right);
            }
        }
    }
    
    for (uint64_t i = CONST_VALUE_2; i <= std::min(num, numMax); i *= CONST_VALUE_2) {
        if (std::find(resList.begin(), resList.end(), i) == resList.end()) {
            resList.emplace_back(i);
        }
    }
    sort(resList.begin(), resList.end());
}

void CalcCommFactor(const uint64_t num, const uint64_t numMax, std::vector<uint64_t> &resList)
{
    uint64_t sqrtMax = static_cast<uint64_t>(sqrt(num));
    for (uint64_t i = 1; i <= sqrtMax; ++i) {
        if (num % i == 0) {
            if (i <= numMax) {
                resList.emplace_back(i);
            }
            uint64_t right = num / i;
            if (right != i && right <= numMax) {
                resList.emplace_back(right);
            }
        }
    }
    sort(resList.begin(), resList.end());
}

void CalcFactorPointWise(uint64_t numMax, std::vector<uint64_t> &resList)
{
    numMax = numMax < CONST_VALUE_2 ? CONST_VALUE_2 : numMax;
    for (uint64_t i = CONST_VALUE_2; i <= numMax; i = i + CONST_VALUE_2) {
        resList.emplace_back(i);
    }
    sort(resList.begin(), resList.end());
}

void VectorElementMultip(std::vector<uint64_t>& range, const uint64_t value)
{
    for (auto& factor : range) {
        factor *= value;
    }
}

bool IsArrayEqual(const std::vector<ConvCommonApi::ConvDtype>& arr1, const std::vector<ConvCommonApi::ConvDtype>& arr2, uint32_t size)
{
    if (arr1.size() < size || arr2.size() < size) {
        return false;
    }
    for (size_t i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

uint64_t InferHiL1(uint64_t inputHoL1, uint64_t hi, uint64_t singlekH, uint32_t dilationH, uint32_t strideH)
{
    uint64_t khDilated = (singlekH - 1) * dilationH + 1;
    uint64_t tmpHiL1 = (inputHoL1 - 1) * strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }

    return tmpHiL1;
}
} // namespace Conv3dTilingApi
