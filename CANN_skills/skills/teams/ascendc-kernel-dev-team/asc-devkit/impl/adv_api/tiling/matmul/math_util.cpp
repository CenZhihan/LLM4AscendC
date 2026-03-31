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
 * \file math_util.cpp
 * \brief
 */

#include "math_util.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace matmul_tiling {
constexpr static int32_t SEED_MAP_MIN = 16;
constexpr static int32_t SEED_MAP_MAX = 1024;
constexpr static int32_t FACTOR_NUM_LIMIT = 4;
constexpr static int32_t L0_FACTOR_NUM_LIMIT = 2;
constexpr static int32_t L1_FACTOR_NUM_LIMIT = 4;
constexpr static int32_t MIN_FACTOR_LIMIT = 8;
constexpr static int32_t L0_FACTOR_LIMIT = 64;
constexpr static int32_t L1_FACTOR_LIMIT = 128;

bool MathUtil::IsEqual(float leftValue, float rightValue)
{
    return std::fabs(leftValue - rightValue) <= std::numeric_limits<float>::epsilon();
}

int32_t MathUtil::AlignDown(int32_t num1, int32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 / num2) * num2;
}

bool MathUtil::CheckMulOverflow(int32_t a, int32_t b, int32_t &c)
{
    if (a > 0 && b > 0) {
        if (a > (INT32_MAX / b)) {
            return false;
        }
    } else {
        return false;
    }
    c = a * b;
    return true;
}

int32_t MathUtil::MapShape(int32_t shape, bool roundUpFlag)
{
    // map numbers between 32 to 1024 to number of power of 2, and map numbers greater than 1024 to 1024.
    uint32_t seed = static_cast<uint32_t>(SEED_MAP_MIN);
    if (shape < static_cast<int32_t>(seed)) {
        return shape;
    }
    while (static_cast<int32_t>(seed) < SEED_MAP_MAX) {
        if (static_cast<int32_t>(seed) < shape && static_cast<int32_t>(seed << 1U) >= shape) {
            break;
        }
        seed = seed << 1U;
    }
    if (roundUpFlag) {
        seed = seed << 1U;
    }
    return static_cast<int32_t>(seed);
}

void MathUtil::GetFactors(std::vector<int32_t> &factorList, int32_t srcNum, int32_t minFactor, int32_t maxFactor)
{
    for (int32_t factor = maxFactor; factor >= minFactor; factor--) {
        if (srcNum % factor == 0) {
            factorList.push_back(factor);
        }
    }
}

void MathUtil::GetFactors(std::vector<int32_t> &factorList, int32_t srcNum, int32_t maxFactor)
{
    int32_t maxNum = std::min(srcNum, maxFactor);
    for (int32_t factor = 1; factor <= maxNum; factor++) {
        if (srcNum % factor == 0) {
            factorList.push_back(factor);
        }
    }
}

void MathUtil::GetFactorCnt(const int32_t shape, int32_t &factorCnt, const int32_t factorStart, const int32_t factorEnd)
{
    for (int32_t i = factorStart; i <= factorEnd; i++) {
        if (shape < i) {
            return;
        }
        if (shape % i == 0) {
            ++factorCnt;
        }
    }
}

void MathUtil::GetFactorLayerCnt(const int32_t shape, int32_t &factorCnt, const int32_t factorStart,
    const int32_t factorEnd)
{
    std::vector<int32_t> factorList;
    MathUtil::GetFactors(factorList, shape, factorStart, factorEnd);
    for (const auto factor : factorList) {
        int32_t fcnt = 0;
        GetFactorCnt(factor, fcnt, 1, factor + 1);
        factorCnt = fcnt >= factorCnt ? fcnt : factorCnt;
    }
}

void MathUtil::AddFactor(std::vector<int32_t> &dimsFactors, int32_t dim)
{
    dimsFactors.push_back(dim);
    sort(dimsFactors.begin(), dimsFactors.end());
    (void)dimsFactors.erase(unique(dimsFactors.begin(), dimsFactors.end()), dimsFactors.cend());
}

int32_t MathUtil::GetNonFactorMap(std::vector<int32_t> &factorList, int32_t srcNum, int32_t maxFactor)
{
    int32_t factorCnt = 0;
    int32_t mapFactor = srcNum;
    MathUtil::GetFactorLayerCnt(srcNum, factorCnt, 1, maxFactor);
    if (srcNum > 1 && factorCnt <= FACTOR_NUM_LIMIT) {
        mapFactor = MathUtil::MapShape(srcNum, true);
    }
    GetFactors(factorList, mapFactor, maxFactor);
    return mapFactor;
}

void MathUtil::GetBlockFactors(std::vector<int32_t> &factorList, const int32_t oriShape, const int32_t mpShape,
    const int32_t coreNum, const int32_t maxNum)
{
    // get all factors of ori_shape/mapshape/coreNum which smaller or equal to maxNum
    for (int32_t i = 1; i <= maxNum; ++i) {
        if ((oriShape % i == 0) || (mpShape % i == 0) || (coreNum % i == 0)) {
            factorList.push_back(i);
        }
    }
}

bool MathUtil::CheckFactorNumSatisfy(const int32_t dim)
{
    if (dim <= MIN_FACTOR_LIMIT) {
        return true;
    }
    int32_t factorL0Cnt = 0;
    int32_t factorL1Cnt = 0;
    MathUtil::GetFactorLayerCnt(dim, factorL0Cnt, 1, L0_FACTOR_LIMIT);
    if (dim > L1_FACTOR_LIMIT) {
        MathUtil::GetFactorLayerCnt(dim, factorL1Cnt, L0_FACTOR_LIMIT + 1, L1_FACTOR_LIMIT);
    }
    bool factorNumNotSatisfied = (factorL0Cnt <= L0_FACTOR_NUM_LIMIT) ||
        ((dim > L1_FACTOR_LIMIT) && (factorL0Cnt + factorL1Cnt <= L1_FACTOR_NUM_LIMIT));
    return !factorNumNotSatisfied;
}

int32_t MathUtil::FindBestSingleCore(const int32_t oriShape, const int32_t mappedShape, const int32_t coreNum,
    bool isKDim)
{
    int32_t bestSingleCore = oriShape;
    int32_t realSingleCore = MathUtil::CeilDivision(oriShape, coreNum);
    int32_t mappedSingleCore = MathUtil::CeilDivision(mappedShape, coreNum);

    if (isKDim) {
        int32_t bestShape = (oriShape % coreNum == 0) ? oriShape : mappedShape;
        bestSingleCore = MathUtil::CeilDivision(bestShape, coreNum);
        return bestSingleCore;
    }

    if (coreNum == 1 && CheckFactorNumSatisfy(bestSingleCore)) {
        return bestSingleCore;
    }

    bestSingleCore = realSingleCore;
    while (bestSingleCore != mappedSingleCore) {
        if (CheckFactorNumSatisfy(bestSingleCore)) {
            return bestSingleCore;
        }
        if (bestSingleCore < mappedSingleCore) {
            ++bestSingleCore;
        } else {
            --bestSingleCore;
        }
    }
    return bestSingleCore;
}

std::vector<std::pair<int32_t, int32_t>> MathUtil::GetFactorPairs(int32_t num)
{
    std::vector<std::pair<int32_t, int32_t>> factors;
    for (int32_t i = 1; i <= num; ++i) {
        if (num % i == 0) {
            factors.emplace_back(i, num / i);
        }
    }
    return factors;
}

std::pair<int32_t, int32_t> MathUtil::DivideIntoMainAndTail(int32_t num, int32_t divisor)
{
    if (divisor == 0) {
        return {0, 0};
    }
    int mainChunk = num / divisor;
    int tailChunk = num % divisor;
    return (tailChunk != 0) ? std::make_pair(mainChunk, tailChunk) : std::make_pair(mainChunk, 0);
}
} // namespace matmul_tiling
