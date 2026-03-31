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
 * \file matmul_tiling_algorithm.cpp
 * \brief
 */
#include "matmul_tiling_algorithm.h"

#include <iostream>
#include <map>
#include <algorithm>
#include <cstring>
#include <cmath>

#include "securec.h"

#include "../../detail/host_log.h"
#include "math_util.h"

using namespace std;

namespace matmul_tiling {
constexpr uint32_t IDX_ZERO = 0;
constexpr uint32_t IDX_ONE = 1;
constexpr uint32_t IDX_TWO = 2;
constexpr uint32_t IDX_THREE = 3;
constexpr uint32_t IDX_FOUR = 4;
constexpr uint32_t IDX_FIVE = 5;
constexpr uint32_t IDX_SIX = 6;
constexpr uint32_t IDX_SEVEN = 7;
constexpr int32_t MAX_BIAS_N = 16;
constexpr int32_t MTE1_L0A_BANDWIDTH = 256;
constexpr int32_t MTE1_L0B_BANDWIDTH = 128;
constexpr int32_t INPUTDTYPE_BYTES = 2;
constexpr int32_t MIN_MTE1_LOAD = 32;
constexpr int32_t BLOCK_CUBE = 16;
constexpr int32_t REDUCE_BLOCK_SIZE = 16;
constexpr int32_t INT8_REDUCE_BLOCK_SIZE = 32;
constexpr int32_t INT4_REDUCE_BLOCK_SIZE = 64;
constexpr int32_t FLOAT32_REDUCE_BLOCK_SIZE = 8;
constexpr int32_t MIN_FRACTAL_SIZE = C0_SIZE * REDUCE_BLOCK_SIZE;
constexpr uint32_t BEST_VALUE_LENGTH = 13;
constexpr int32_t BEST_VALUE_LIST[BEST_VALUE_LENGTH] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
constexpr uint32_t DIM_FACTOR_LENGTH = 4;
constexpr uint64_t UINT64_TYPES = 8;
constexpr int32_t L0B_ALIGN_SIZE = 2;

// minimum factor number requirement for the data amount in single-core
constexpr int32_t L0_FACTOR_NUM_LIMIT = 2;
constexpr int32_t L1_FACTOR_NUM_LIMIT = 4;
// the lower bound of the factor number check
constexpr int32_t L0_FACTOR_LIMIT = 64;
constexpr int32_t L1_FACTOR_LIMIT = 128;
constexpr int32_t L0_SIZE = 64 * 1024;

static MatmulTemplateCfg g_tempCfg;

constexpr int32_t MTE1_FIXPIPE_BANDWIDTH = 128; // 128 Bytes per cycle

constexpr int32_t NUM_TWO = 2;

constexpr int32_t ATTACH_FLAG_ZERO = 0;
constexpr int32_t ATTACH_FLAG_ONE = 1;
constexpr int32_t ATTACH_FLAG_TWO = 2;

constexpr int32_t INT8_ALIGN_SIZE = 32;
constexpr int32_t FP32_ALIGN_SIZE = 16;
constexpr int32_t INT4_ALIGN_SIZE = 64;
constexpr int32_t DATA_COPY_ALIGN_SIZE = 256; // 256B

constexpr int DT_FLOAT_INVALID_BASEK = 8;

constexpr int32_t MX_BASEK_FACTOR = 64;
constexpr int32_t MX_L1_BUFFER_NUM = 4;
constexpr int32_t MX_L1_TO_L0_ALIGN = 1024;
constexpr uint32_t SCALE_FACTOR_MAX_VALUE = 127;
constexpr int32_t SCALE_K_SIZE = 32;
constexpr uint64_t SCALE_FACTOR_A_MASK = 0x0000007f;
constexpr uint64_t SCALE_FACTOR_B_MASK = 0x00007f00;
constexpr uint32_t SCALE_FACTOR_B_OFFSET = 8;
constexpr int32_t BASIC_SIZE_32 = 32;
constexpr int32_t BAND_LIMIT_MAX_CORENUM = 20;
constexpr float MAX_BAND_WIDTH_RATIO = 4;
const std::map<uint32_t, float> BAND_WIDTH_TAB = { // (numcols, band ratio)
    {256, 1}, {512, 1}, {384, 1.5}, {448, 1.7},
    {320, 2}, {128, 2}, {192, 2.7}
};

constexpr int32_t N_BUFFER_33_FACTOR = 3;

namespace {
bool IsOrgShapeAlign(int32_t orgShape, int32_t alignSize, bool isSmallShape = false)
{
    if (alignSize == 0) {
        return false;
    }
    if (!isSmallShape) {
        return orgShape % alignSize == 0;
    }
    for (const auto& pair : BAND_WIDTH_TAB) {
        uint32_t value = pair.first;
        if (value == 0) {
            continue;
        }
        if (orgShape % value == 0) {
            return true;
        }
    }
    return false;
}
}

MatmulTilingAlgorithm::MatmulTilingAlgorithm(MatmulApiTilingBase* tilingIns)
{
    ASCENDC_HOST_ASSERT(tilingIns != nullptr, {}, "tiling instance is null");
    tilingIns_ = tilingIns;
}

int32_t MatmulTilingAlgorithm::GetC0Size() const
{
    if (tilingIns_->aType_.dataType == DataType::DT_FLOAT) {
        return FLOAT32_REDUCE_BLOCK_SIZE;
    } else if (tilingIns_->aType_.dataType == DataType::DT_INT8) {
        return INT8_REDUCE_BLOCK_SIZE;
    } else if (tilingIns_->aType_.dataType == DataType::DT_INT4) {
        return INT4_REDUCE_BLOCK_SIZE;
    }
    return REDUCE_BLOCK_SIZE;
}

int32_t MatmulTilingAlgorithm::GetBestValue(int32_t base) const
{
    for (uint32_t i = 0; i < BEST_VALUE_LENGTH; ++i) {
        if (i == 0 || BEST_VALUE_LIST[i] <= base) {
            continue;
        }
        return BEST_VALUE_LIST[i - 1];
    }
    return BEST_VALUE_LIST[BEST_VALUE_LENGTH - 1];
}

void MatmulTilingAlgorithm::GetTwoFactors(int32_t (&res)[2], int32_t base, int32_t dim, int32_t maxNum) const
{
    if (dim == 1) {
        res[0] = 1;
        res[1] = 1;
        return;
    }

    res[0] = 0;
    res[1] = 0;

    int cnt = 0;
    for (auto up = base + 1; up <= maxNum && up <= dim; ++up) {
        if (dim % up == 0) {
            res[cnt++] = up;
            break;
        }
    }

    for (auto down = base; down >= 1; --down) {
        if (dim % down == 0) {
            res[cnt++] = down;
            if (cnt == sizeof(res) / sizeof(res[0])) {
                break;
            }
        }
    }
}

void MatmulTilingAlgorithm::GetABL1KAlignValue(int32_t& kaAlignValue, int32_t& kbAlignValue) const
{
    kaAlignValue = 1;
    kbAlignValue = 1;
    if (tilingIns_->aType_.dataType == DataType::DT_FLOAT || tilingIns_->bType_.dataType == DataType::DT_FLOAT) {
        // when in FP32 mode, k_a must be an even number if k-alignment is needed. So make ka_align_value as 2.
        kaAlignValue = tilingIns_->aType_.isTrans ? 2 : 1;
        // Same as previous one, make kb_align_value as 2 when k-alignment is needed
        kbAlignValue = (tilingIns_->aType_.isTrans || !tilingIns_->bType_.isTrans) ? 2 : 1;
    }
}

void MatmulTilingAlgorithm::GetL0StatusFromParasCombo(L0StatusPack& l0Status, int32_t* parasCombo) const
{
    l0Status.InitLoadStatus();
    size_t kIdx = 0;
    l0Status.dbL0A = parasCombo[kIdx++];
    l0Status.dbL0B = parasCombo[kIdx++];
    l0Status.dbL0C = parasCombo[kIdx++];
    l0Status.maxMk = parasCombo[kIdx++];
    l0Status.maxNk = parasCombo[kIdx++];
    l0Status.maxMn = parasCombo[kIdx++];
    l0Status.maxAxisIdx = parasCombo[kIdx++];
    l0Status.maxAxisNum = parasCombo[kIdx++];
    l0Status.maxAxisPnt = parasCombo[kIdx++];
    l0Status.maxN = parasCombo[kIdx++];
    l0Status.maxAxisPnt = min(l0Status.maxAxisPnt, l0Status.maxAxisNum);
}

void MatmulTilingAlgorithm::SetResFactors(L0Factors& resFactors, const L0StatusPack& l0Status) const
{
    resFactors.finalML0 = l0Status.finalML0;
    resFactors.finalKL0 = l0Status.finalKL0;
    resFactors.finalNL0 = l0Status.finalNL0;
    resFactors.finalLoadSize = l0Status.finalLoadSize;
    resFactors.finalL0cUse = l0Status.finalL0cUse;
    resFactors.finalMte1Loop = l0Status.finalMte1Loop;
    resFactors.finalMul = l0Status.finalMul;
    resFactors.finalMte1Cycles = l0Status.finalMte1Cycles;
}

int32_t MatmulTilingAlgorithm::GetLoadSize(const CoreStatusPack& coreStatus, const L0StatusPack& l0Status) const
{
    const bool al0FullLoad =
        (static_cast<int64_t>(coreStatus.m * coreStatus.k) * static_cast<int64_t>(C0_SIZE * C0_BYTE_SIZE)) <=
        static_cast<int64_t>(tilingIns_->bufferPool_.l0ASize);
    const bool bl0FullLoad =
        (static_cast<int64_t>(coreStatus.n * coreStatus.k) * static_cast<int64_t>(C0_SIZE * C0_BYTE_SIZE)) <=
        static_cast<int64_t>(tilingIns_->bufferPool_.l0BSize);
    const bool kFullLoad = (l0Status.kL0 >= coreStatus.k);
    if (al0FullLoad || bl0FullLoad) {
        return coreStatus.m + coreStatus.n;
    } else if (kFullLoad) {
        return min(coreStatus.n + MathUtil::CeilDivision(coreStatus.n, l0Status.nL0) * coreStatus.m,
                   coreStatus.m + MathUtil::CeilDivision(coreStatus.m, l0Status.mL0) * coreStatus.n);
    } else {
        return MathUtil::CeilDivision(coreStatus.m, l0Status.mL0) * coreStatus.n +
               MathUtil::CeilDivision(coreStatus.n, l0Status.nL0) * coreStatus.m;
    }
}

bool MatmulTilingAlgorithm::CheckBaseMNKL1Size(SingleCoreStatus& singleCoreStatus) const
{
    L0StatusPack& l0Status = singleCoreStatus.l0Status;
    int32_t a1Length = static_cast<int32_t>(l0Status.mL0 * l0Status.kL0 * C0_SIZE * C0_BYTE_SIZE);
    int32_t b1Length = static_cast<int32_t>(l0Status.nL0 * l0Status.kL0 * C0_SIZE * C0_BYTE_SIZE);
    int32_t biasLength = (tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND910B ||
                          tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND310B)
                             ? l0Status.nL0 * C0_SIZE * DTYPE_BYTE_TAB.at(tilingIns_->biasType_.dataType)
                             : 0;
    int32_t dequantSize = 0;
    if (tilingIns_->deqType == DequantType::TENSOR) {
        dequantSize = l0Status.nL0 * C0_SIZE * UINT64_TYPES;
    }
    if (tilingIns_->aType_.pos == TPosition::TSCM) {
        a1Length = 0;
    }
    if (tilingIns_->bType_.pos == TPosition::TSCM) {
        b1Length = 0;
    }
    // Only V220/V300 bias uses L1 space.
    if (tilingIns_->biasType_.pos == TPosition::TSCM || !tilingIns_->isBias) {
        biasLength = 0;
    }
    const int32_t totalLength = a1Length + b1Length + biasLength + dequantSize;
    return totalLength <= tilingIns_->bufferPool_.l1Size;
}

bool MatmulTilingAlgorithm::CheckK0Align(int32_t k0) const
{
    if ((tilingIns_->aType_.dataType == DataType::DT_FLOAT && tilingIns_->aType_.type == CubeFormat::NZ &&
        tilingIns_->aType_.isTrans) ||
        (tilingIns_->bType_.dataType == DataType::DT_FLOAT && tilingIns_->bType_.type == CubeFormat::NZ &&
        !tilingIns_->bType_.isTrans)) {
        return k0 % NUM_TWO == 0;
    }
    return true;
}

void MatmulTilingAlgorithm::GetFinalMkn(SingleCoreStatus& singleCoreStatus, const CoreStatusPack& coreStatus,
    const int32_t& k0, const int32_t& majorDimFactor, const int32_t& minorDimFactor) const
{
    if (k0 == 0) {
        return;
    }
    L0StatusPack& l0Status = singleCoreStatus.l0Status;
    if (l0Status.maxAxisIdx == 0) {
        l0Status.mL0 = majorDimFactor;
        l0Status.nL0 = minorDimFactor;
    } else {
        l0Status.mL0 = minorDimFactor;
        l0Status.nL0 = majorDimFactor;
    }
    l0Status.kL0 = k0;
    const float tmpL0cUse = static_cast<float>(l0Status.mL0 * l0Status.nL0 * l0Status.dbL0C *
        MIN_FRACTAL_SIZE * FP32_BYTES * 1.0 / tilingIns_->bufferPool_.l0CSize);
    // NUM_TWO means L0A and L0B double buffer is default-on.

    const int32_t tmpMte1Cycle =
        max(2 * 3, l0Status.mL0 * l0Status.kL0 * C0_SIZE * C0_BYTE_SIZE / MTE1_L0A_BANDWIDTH) +
        max(2 * 3, l0Status.kL0 * l0Status.nL0 * C0_SIZE * C0_BYTE_SIZE / MTE1_L0B_BANDWIDTH);
    const int32_t tmpMadCycle = l0Status.mL0 * l0Status.kL0 * l0Status.nL0; // (m<=4 or n<=2:tmpMte1Cycle > tmpMadCycle)
    const int32_t tmpLoadSize = GetLoadSize(coreStatus, l0Status);
    // calculate load2d loop: A splitK for K loop; B split K for m loop as to V100
    const int32_t tmpMte1Loop = ((l0Status.nL0 != 1) ? l0Status.kL0 : 1) + ((l0Status.kL0 != 1) ? l0Status.mL0 : 1);

    const bool condition1 = l0Status.finalML0 == 0;
    const bool condition2 =
        (tmpLoadSize < l0Status.finalLoadSize) || (tmpMte1Cycle < tmpMadCycle && !l0Status.updateUsingMte1);
    const bool condition3 = (tmpLoadSize == l0Status.finalLoadSize && tmpMadCycle > l0Status.finalMul &&
        tmpMadCycle * tmpL0cUse >= l0Status.finalMul * l0Status.finalL0cUse);
    const bool condition4 = tmpMadCycle == l0Status.finalMul && tmpLoadSize == l0Status.finalLoadSize &&
        tmpMte1Loop < l0Status.finalMte1Loop;
    // Considering pipeline parallelism between MTE1 and MAD
    const bool condition5 = ((tmpMte1Cycle < tmpMadCycle && l0Status.updateUsingMte1) || !l0Status.updateUsingMte1);
    const bool condition6 = CheckBaseMNKL1Size(singleCoreStatus);
    int32_t lastReduceDim = (tilingIns_->aType_.dataType == DataType::DT_FLOAT ||
        tilingIns_->bType_.dataType == DataType::DT_FLOAT) ? FLOAT32_REDUCE_BLOCK_SIZE : REDUCE_BLOCK_SIZE;

    const bool condition7 = (tilingIns_->baseN != -1) || (!(coreStatus.n >= lastReduceDim && l0Status.nL0 <
        lastReduceDim));

    const bool condition8 = CheckK0Align(l0Status.kL0);

    const bool validL0 = (condition1 || condition2 || condition3 || condition4) && condition5 &&
        condition6 && condition7 && condition8;
    if (validL0) {
        l0Status.finalML0 = l0Status.mL0;
        l0Status.finalKL0 = l0Status.kL0;
        l0Status.finalNL0 = l0Status.nL0;
        l0Status.finalLoadSize = tmpLoadSize;
        l0Status.finalL0cUse = tmpL0cUse;
        l0Status.finalMul = tmpMadCycle;
        l0Status.finalMte1Cycles = tmpMte1Cycle;
        l0Status.finalMte1Loop = tmpMte1Loop;
        l0Status.updateUsingMte1 = l0Status.updateUsingMte1 || (tmpMte1Cycle < tmpMadCycle);
    }
}

void MatmulTilingAlgorithm::GetL0bAlign(std::vector<int32_t>& factors) const
{
    constexpr int32_t alignSize = 2;
    if (DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8) ||
        DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT4)) {
        for (auto& num : factors) {
            num = MathUtil::Align(num, alignSize);
        }
    }
    return;
}

void MatmulTilingAlgorithm::GetL0FactorsCand(L0Factors& resFactors, const CoreStatusPack& coreStatus,
    SingleCoreStatus& singleCoreStatus, int32_t* parasCombo, const MatmulRunParas& param) const
{
    (void)(param);
    L0StatusPack& l0Status = singleCoreStatus.l0Status;
    GetL0StatusFromParasCombo(l0Status, parasCombo);
    int32_t l0bAlignSize = 1;
    if (DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8) ||
        DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT4)) {
        l0bAlignSize = L0B_ALIGN_SIZE;
    }
    int32_t majorDim = coreStatus.m;
    int32_t minorDim = MathUtil::Align(coreStatus.n, l0bAlignSize);
    int32_t majorDimK = l0Status.maxMk;
    int32_t minorDimK = l0Status.maxNk;
    int32_t maxN = l0Status.maxN;
    int32_t dimFactors[2] = {0};
    if (l0Status.maxAxisIdx != 0) {
        majorDim = MathUtil::Align(coreStatus.n, l0bAlignSize);
        minorDim = coreStatus.m;
        majorDimK = l0Status.maxNk;
        minorDimK = l0Status.maxMk;
    }

    std::vector<int32_t> majorDimFactors(DIM_FACTOR_LENGTH, 0);
    if (tilingIns_->baseN != -1 && l0Status.maxAxisIdx != 0) {
        majorDimFactors[0] = MathUtil::CeilDivision(tilingIns_->baseN, C0_SIZE);
    } else if (tilingIns_->baseM != -1 && l0Status.maxAxisIdx == 0) {
        majorDimFactors[0] = MathUtil::CeilDivision(tilingIns_->baseM, C0_SIZE);
    } else {
        // n dim condition
        if (l0Status.maxAxisIdx != 0 && tilingIns_->isSupportL0c2Out && tilingIns_->isBias) {
            GetTwoFactors(dimFactors, min(l0Status.maxAxisPnt, maxN), majorDim, min(l0Status.maxAxisNum, maxN));
        } else {
            GetTwoFactors(dimFactors, l0Status.maxAxisPnt, majorDim, l0Status.maxAxisNum);
        }
        majorDimFactors[0] = dimFactors[0];
        majorDimFactors[1] = dimFactors[1];
        const int32_t majorAmend = GetBestValue(majorDim);
        if (l0Status.maxAxisIdx != 0 && tilingIns_->isSupportL0c2Out && tilingIns_->isBias) {
            GetTwoFactors(dimFactors, min(l0Status.maxAxisPnt, maxN), majorAmend, min(l0Status.maxAxisNum, maxN));
        } else {
            GetTwoFactors(dimFactors, l0Status.maxAxisPnt, majorAmend, l0Status.maxAxisNum);
        }
        majorDimFactors[IDX_TWO] = dimFactors[0];
        majorDimFactors[IDX_THREE] = dimFactors[1];
        if (l0Status.maxAxisIdx != 0) {
            GetL0bAlign(majorDimFactors);
        }
    }
    sort(majorDimFactors.rbegin(), majorDimFactors.rend());
    majorDimFactors.erase(unique(majorDimFactors.begin(), majorDimFactors.end()), majorDimFactors.end());
    for (auto& majorDimFactor : majorDimFactors) {
        if (majorDimFactor == 0 || majorDimFactor > l0Status.maxMn || majorDimFactor > majorDimK ||
            majorDimFactor > majorDim) {
            continue;
        }
        const int32_t minorFactorMax = min(l0Status.maxMn / majorDimFactor, minorDimK);
        std::vector<int32_t> minorDimFactors(DIM_FACTOR_LENGTH, 0);
        if (tilingIns_->baseN != -1 && l0Status.maxAxisIdx == 0) {
            minorDimFactors[0] = MathUtil::CeilDivision(tilingIns_->baseN, C0_SIZE);
        } else if (tilingIns_->baseM != -1 && l0Status.maxAxisIdx != 0) {
            minorDimFactors[0] = MathUtil::CeilDivision(tilingIns_->baseM, C0_SIZE);
        } else {
            if (l0Status.maxAxisIdx == 0 && tilingIns_->isSupportL0c2Out && tilingIns_->isBias) {
                GetTwoFactors(dimFactors, min(minorFactorMax, maxN), minorDim, min(minorFactorMax, maxN));
            } else {
                GetTwoFactors(dimFactors, minorFactorMax, minorDim, minorFactorMax);
            }
            minorDimFactors[0] = dimFactors[0];
            minorDimFactors[1] = dimFactors[1];
            const int32_t minorAmend = GetBestValue(minorDim);
            if (l0Status.maxAxisIdx == 0 && tilingIns_->isSupportL0c2Out && tilingIns_->isBias) {
                GetTwoFactors(dimFactors, min(minorFactorMax, maxN), minorAmend, min(minorFactorMax, maxN));
            } else {
                GetTwoFactors(dimFactors, minorFactorMax, minorAmend, minorFactorMax);
            }
            minorDimFactors[IDX_TWO] = dimFactors[0];
            minorDimFactors[IDX_THREE] = dimFactors[1];
            if (l0Status.maxAxisIdx == 0) {
                GetL0bAlign(minorDimFactors);
            }
        }
        sort(minorDimFactors.rbegin(), minorDimFactors.rend());
        minorDimFactors.erase(unique(minorDimFactors.begin(), minorDimFactors.end()), minorDimFactors.end());
        for (auto& minorDimFactor : minorDimFactors) {
            if (minorDimFactor == 0 || minorDimFactor * majorDimFactor > l0Status.maxMn || minorDimFactor > minorDimK ||
                (minorDimFactor > minorDim) || (minorDimFactor > majorDimK)) {
                continue;
            }
            // consider bias table buffer
            constexpr int32_t maxN0 = 64;
            // in V220/V300, consider bias table buffer limit
            if (tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND910B ||
                tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND310B) {
                maxN = tilingIns_->bufferPool_.btSize / C0_SIZE / FP32_BYTES / l0Status.dbL0C;
            }
            if (l0Status.maxAxisIdx != 0) {
                // Major is n0 axis
                if ((majorDimFactor > maxN0) && tilingIns_->isSupportL0c2Out && tilingIns_->isBias) {
                    continue;
                }
            } else {
                // Major is m0  axis
                if ((minorDimFactor > maxN0) && tilingIns_->isSupportL0c2Out && tilingIns_->isBias) {
                    continue;
                }
            }

            const int32_t k0Max = min(majorDimK / majorDimFactor, minorDimK / minorDimFactor);
            std::vector<int32_t> k0Factors(DIM_FACTOR_LENGTH, 0);
            GetTwoFactors(dimFactors, k0Max, coreStatus.k, k0Max);
            k0Factors[0] = dimFactors[0];
            k0Factors[1] = dimFactors[1];
            const int32_t kAmend = GetBestValue(coreStatus.k);
            GetTwoFactors(dimFactors, k0Max, kAmend, l0Status.maxAxisNum);
            k0Factors[IDX_TWO] = dimFactors[0];
            k0Factors[IDX_THREE] = dimFactors[1];
            sort(k0Factors.rbegin(), k0Factors.rend());
            k0Factors.erase(unique(k0Factors.begin(), k0Factors.end()), k0Factors.end());
            for (auto& k0 : k0Factors) {
                if (k0 == 0 || minorDimFactor * k0 > minorDimK || majorDimFactor * k0 > majorDimK) {
                    continue;
                }
                // when k_axis <= 8, adjust k0_factor to 2, to generate tiling baseK align to 16
                if (((tilingIns_->aType_.dataType == DataType::DT_FLOAT && tilingIns_->aType_.type == CubeFormat::NZ &&
                    tilingIns_->aType_.isTrans) ||
                    (tilingIns_->bType_.dataType == DataType::DT_FLOAT && tilingIns_->bType_.type == CubeFormat::NZ &&
                    !tilingIns_->bType_.isTrans)) && k0 == 1) {
                    k0 = NUM_TWO;
                }
                // Check if the buffer size allocated exceed the hardware buffer size in Float Mode
                if (tilingIns_->aType_.dataType == DataType::DT_FLOAT) {
                    int32_t mL0 = majorDimFactor;
                    int32_t nL0 = minorDimFactor;
                    if (l0Status.maxAxisIdx != 0) {
                        nL0 = majorDimFactor;
                        mL0 = minorDimFactor;
                    }

                    const int32_t l0aBufferSize = tilingIns_->aType_.isTrans ?
                        MathUtil::Align(k0, 2) * C0_BYTE_SIZE * mL0 * C0_SIZE * DB_ON :
                        k0 * C0_BYTE_SIZE * mL0 * C0_SIZE * DB_ON;
                    const int32_t l0bBufferSize = (tilingIns_->aType_.isTrans || !tilingIns_->bType_.isTrans) ?
                        MathUtil::Align(k0, 2) * C0_BYTE_SIZE * nL0 * C0_SIZE * DB_ON :
                        k0 * C0_BYTE_SIZE * nL0 * C0_SIZE * DB_ON;
                    if (l0aBufferSize > tilingIns_->bufferPool_.l0ASize || l0bBufferSize > tilingIns_->bufferPool_.l0BSize) {
                        continue;
                    }
                } else if (DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8) ||
                            DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT4)) {
                    int32_t mL0 = majorDimFactor;
                    int32_t nL0 = minorDimFactor;
                    if (l0Status.maxAxisIdx != 0) {
                        nL0 = majorDimFactor;
                        mL0 = minorDimFactor;
                    }

                    const int32_t l0aBufferSize = tilingIns_->aType_.isTrans ?
                        k0 * C0_BYTE_SIZE * MathUtil::Align(mL0, 2) * C0_SIZE * DB_ON :
                        k0 * C0_BYTE_SIZE * mL0 * C0_SIZE * DB_ON;
                    int32_t l0bBufferSize = (tilingIns_->bType_.isTrans) ?
                        k0 * C0_BYTE_SIZE * nL0 * C0_SIZE * DB_ON :
                        k0 * C0_BYTE_SIZE *  MathUtil::Align(nL0, 2) * C0_SIZE * DB_ON;
                    if (l0aBufferSize > tilingIns_->bufferPool_.l0ASize ||
                        l0bBufferSize > tilingIns_->bufferPool_.l0BSize) {
                        continue;
                    }
                }
                GetFinalMkn(singleCoreStatus, coreStatus, k0, majorDimFactor, minorDimFactor);
            }
        }
    }
    if (l0Status.finalML0 != 0 && l0Status.finalKL0 != 0 && l0Status.finalNL0 != 0) {
        SetResFactors(resFactors, l0Status);
    }
}

MKNParasCombo MatmulTilingAlgorithm::GetParasCombo(const int32_t& index, const MatmulRunParas& param) const
{
    (void)(param);
    std::map<int32_t, MKNParasCombo> parasComboMap;
    const int32_t mnMax = tilingIns_->bufferPool_.l0CSize / (C0_SIZE * C0_SIZE) / FP32_BYTES;
    int32_t maxN = 64;
    // in V220/V300, consider bias table buffer limit
    if (tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND910B ||
        tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND310B) {
        maxN = tilingIns_->bufferPool_.btSize / C0_SIZE / FP32_BYTES;
    }
    const bool biasBt = tilingIns_->isSupportL0c2Out && (tilingIns_->isBias);
    const int32_t leftSize = min(tilingIns_->bufferPool_.l1Size, tilingIns_->bufferPool_.l0ASize / dbL0A_);
    const int32_t rightSize = min(tilingIns_->bufferPool_.l1Size, tilingIns_->bufferPool_.l0BSize / dbL0B_);
    const int32_t maxMk = tilingIns_->aType_.pos == TPosition::TSCM ? 64 : (leftSize / C0_SIZE / C0_BYTE_SIZE);
    const int32_t maxNK = tilingIns_->bType_.pos == TPosition::TSCM ? 64 : (rightSize / C0_SIZE / C0_BYTE_SIZE);
    // dbL0A, dbL0B, dbL0C, maxMk, maxNk, maxMn, maxAxisIdx, maxAxisNum, maxAxisPnt, maxN
    MKNParasCombo comboZero = { 2, 2, 2, maxMk, maxNK, mnMax / DB_ON, 0, 64, 8, biasBt ? maxN / DB_ON : 64 };
    MKNParasCombo comboOne = { dbL0A_, dbL0B_, 1, maxMk, maxNK, mnMax, 0, 64, 11, biasBt ? maxN : 64 };
    parasComboMap = { { 0, comboZero }, { 1, comboOne } };

    return parasComboMap[index];
}

void MatmulTilingAlgorithm::GetL0cDB(const L0Factors (&resFactors)[L0PARAS_COMBO_LEN], const CoreStatusPack& coreStatus,
    L0StatusPack& l0Status) const
{
    const int32_t dbAOnBOnCOnIdx = 0;
    const int32_t dbAOnBOnCOffIdx = 1;
    // check both L0C utilization and loadsize to control LOC LOA LOB DB
    const int32_t m0L0cDbOn = resFactors[dbAOnBOnCOnIdx].finalML0;
    const int32_t k0L0cDbOn = resFactors[dbAOnBOnCOnIdx].finalKL0;
    const int32_t n0L0cDbOn = resFactors[dbAOnBOnCOnIdx].finalNL0;
    const int32_t loadSizeL0cDbOn = resFactors[dbAOnBOnCOnIdx].finalLoadSize;
    const int32_t mte1CyclesL0cDbOn = resFactors[dbAOnBOnCOnIdx].finalMte1Cycles;

    const int32_t m0L0cDbOff = resFactors[dbAOnBOnCOffIdx].finalML0;
    const int32_t k0L0cDbOff = resFactors[dbAOnBOnCOffIdx].finalKL0;
    const int32_t n0L0cDbOff = resFactors[dbAOnBOnCOffIdx].finalNL0;
    const int32_t loadSizeL0cDbOff = resFactors[dbAOnBOnCOffIdx].finalLoadSize;
    const int32_t mte1CyclesL0cDbOff = resFactors[dbAOnBOnCOffIdx].finalMte1Cycles;

    const int32_t mte3CostDbOn =
        m0L0cDbOn * n0L0cDbOn * MIN_FRACTAL_SIZE * FP16_BYTES * 1 / MTE1_FIXPIPE_BANDWIDTH;
    const int32_t mte3CostDbOff =
        m0L0cDbOff * n0L0cDbOff * MIN_FRACTAL_SIZE * FP16_BYTES * 1 / MTE1_FIXPIPE_BANDWIDTH;

    const int32_t madCylesDbOn = max(m0L0cDbOn * k0L0cDbOn * n0L0cDbOn, static_cast<int32_t>(mte1CyclesL0cDbOn * 0.7));
    const int32_t madCylesDbOff =
        max(m0L0cDbOff * k0L0cDbOff * n0L0cDbOff, static_cast<int32_t>(mte1CyclesL0cDbOff * 0.7));
    int32_t dbOnPipeTime = MathUtil::CeilDivision(coreStatus.m, m0L0cDbOn) *
        MathUtil::CeilDivision(coreStatus.n, n0L0cDbOn) *
        ((MathUtil::CeilDivision(coreStatus.k, k0L0cDbOn) - 1) * madCylesDbOn + max(madCylesDbOn, mte3CostDbOn));
    int32_t dbOffPipeTime = MathUtil::CeilDivision(coreStatus.m, m0L0cDbOff) *
        MathUtil::CeilDivision(coreStatus.n, n0L0cDbOff) *
        (MathUtil::CeilDivision(coreStatus.k, k0L0cDbOff) * madCylesDbOff + mte3CostDbOff);
    dbOnPipeTime = dbOnPipeTime == 0 ? INT32_MAX : dbOnPipeTime;
    dbOffPipeTime = dbOffPipeTime == 0 ? INT32_MAX : dbOffPipeTime;

    if ((dbOffPipeTime < dbOnPipeTime) || (loadSizeL0cDbOff < loadSizeL0cDbOn)) {
        l0Status.dbL0C = 1;
        l0Status.dbL0A = dbL0A_;
        l0Status.dbL0B = dbL0B_;
        l0Status.mL0 = m0L0cDbOff;
        l0Status.kL0 = k0L0cDbOff;
        l0Status.nL0 = n0L0cDbOff;
    } else {
        l0Status.dbL0C = DB_ON;
        l0Status.dbL0A = dbL0A_;
        l0Status.dbL0B = dbL0B_;
        l0Status.mL0 = m0L0cDbOn;
        l0Status.kL0 = k0L0cDbOn;
        l0Status.nL0 = n0L0cDbOn;
    }
}

int32_t MatmulTilingAlgorithm::GetMxCurL1Size(const SingleCoreStatus& singleCoreStatus) const
{
    int32_t curAL1Size = 0;
    int32_t curBL1Size = 0;
    int32_t curScaleAL1Size = 0;
    int32_t curScaleBL1Size = 0;
    int32_t curBiasSize = 0;
    uint32_t bL1Const = 1;
    uint32_t aL1Const = 1;
    int32_t reduceSize = static_cast<int32_t>(C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);
    if (IsNeedAlign(true)) {
        aL1Const *= reduceSize / C0_SIZE;
    }
    if (IsNeedAlign(false)) {
        bL1Const *= reduceSize / C0_SIZE;
    }
    int32_t baseM = singleCoreStatus.l0Status.mL0 * C0_SIZE * aL1Const;
    int32_t baseN = singleCoreStatus.l0Status.nL0 * C0_SIZE * bL1Const;
    int32_t baseK = MathUtil::Align(singleCoreStatus.l0Status.kL0 * reduceSize, MX_BASEK_FACTOR);

    if (tilingIns_->aType_.pos == TPosition::VECOUT) {
        curAL1Size = GetSingleM() * GetSingleK() * DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
    } else if (tilingIns_->aType_.pos == TPosition::GM) {
        curAL1Size = baseM * baseK * DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
    }

    if (tilingIns_->bType_.pos == TPosition::VECOUT) {
        curBL1Size = GetSingleN() * GetSingleK() * DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
    } else if (tilingIns_->bType_.pos == TPosition::GM) {
        curBL1Size = baseN * baseK * DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
    }

    if (tilingIns_->aType_.scalePos == TPosition::VECOUT) {
        curScaleAL1Size = GetSingleM() * MathUtil::CeilDivision(GetSingleK(), MX_BASEK_FACTOR) * NUM_TWO;
    } else if (tilingIns_->aType_.scalePos == TPosition::GM) {
        curScaleAL1Size = baseM * MathUtil::CeilDivision(baseK, MX_BASEK_FACTOR) * NUM_TWO;
    }

    if (tilingIns_->bType_.scalePos == TPosition::VECOUT) {
        curScaleBL1Size = GetSingleN() * MathUtil::CeilDivision(GetSingleK(), MX_BASEK_FACTOR) * NUM_TWO;
    } else if (tilingIns_->bType_.scalePos == TPosition::GM) {
        curScaleBL1Size = baseN * MathUtil::CeilDivision(baseK, MX_BASEK_FACTOR) * NUM_TWO;
    }

    if (tilingIns_->isBias) {
        if (tilingIns_->biasType_.pos == TPosition::VECOUT) {
            curBiasSize = GetSingleN() * DTYPE_BIT_TAB.at(tilingIns_->biasType_.dataType) / BITS_PER_BYTE;
        } else if (tilingIns_->biasType_.pos == TPosition::GM) {
            curBiasSize = baseN * DTYPE_BIT_TAB.at(tilingIns_->biasType_.dataType) / BITS_PER_BYTE;
        }
    }
    return curAL1Size + curBL1Size + curScaleAL1Size + curScaleBL1Size + curBiasSize;
}

void MatmulTilingAlgorithm::GetL0Factors(const std::string& opType, const MatmulRunParas& param,
    const CoreStatusPack& coreStatus, SingleCoreStatus& singleCoreStatus) const
{
    (void)(opType);
    // get mL0, nL0, kL0 factor when singlecore m, n, k is know
    // mL0, nL0, kL0 is a factor of single core m, n, k
    L0StatusPack& l0Status = singleCoreStatus.l0Status;
    if (tilingIns_->isBias) {
        l0Status.dtypeBias = DTYPE_BYTE_TAB.at(tilingIns_->biasType_.dataType);
    }
    L0Factors resFactors[L0PARAS_COMBO_LEN];
    for (int32_t i = 0; i < L0PARAS_COMBO_LEN; ++i) {
        if (i == 0 && g_tempCfg.l0cDB == DB_OFF) {
            continue;
        }
        MKNParasCombo mknParasCombo = GetParasCombo(i, param);
        for (int32_t j = 0; j < L0PARAS_COMBO_LEN; ++j) {
            mknParasCombo.parasCombo[IDX_SIX] = j;
            GetL0FactorsCand(resFactors[i], coreStatus, singleCoreStatus, mknParasCombo.parasCombo, param);
        }
    }
    if (g_tempCfg.l0cDB == DB_OFF) {
        l0Status.dbL0C = DB_OFF;
        l0Status.dbL0A = dbL0A_;
        l0Status.dbL0B = dbL0B_;
        l0Status.mL0 = resFactors[1].finalML0;
        l0Status.kL0 = resFactors[1].finalKL0;
        l0Status.nL0 = resFactors[1].finalNL0;
    } else {
        GetL0cDB(resFactors, coreStatus, l0Status);
    }

    if (tilingIns_->madType_ == MatrixMadType::MXMODE) {
        int32_t curL1Size = GetMxCurL1Size(singleCoreStatus);
        int32_t reduceSize = static_cast<int32_t>(C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);
        if (tilingIns_->aType_.pos == TPosition::TSCM && tilingIns_->bType_.pos == TPosition::GM) {
            if (curL1Size >= tilingIns_->bufferPool_.l1Size) {
                l0Status.nL0 = (!tilingIns_->bType_.isTrans && tilingIns_->bType_.type == CubeFormat::NZ) ? NUM_TWO : 1;
                l0Status.kL0 = MX_BASEK_FACTOR / reduceSize;
            }
        }

        if (tilingIns_->aType_.pos == TPosition::GM && tilingIns_->bType_.pos == TPosition::TSCM) {
            if (curL1Size >= tilingIns_->bufferPool_.l1Size) {
                l0Status.mL0 = (tilingIns_->aType_.isTrans && tilingIns_->aType_.type == CubeFormat::NZ) ? NUM_TWO : 1;
                l0Status.kL0 = MX_BASEK_FACTOR / reduceSize;
            }
        }
    }
}

bool MatmulTilingAlgorithm::IsNeedAlign(bool isA) const
{
    if (tilingIns_->madType_ != MatrixMadType::MXMODE) {
        if (isA) {
            return tilingIns_->aType_.dataType == DataType::DT_FLOAT || (tilingIns_->aType_.dataType == DataType::DT_INT8 && tilingIns_->aType_.isTrans);
        } else {
            return tilingIns_->bType_.dataType == DataType::DT_FLOAT || (tilingIns_->bType_.dataType == DataType::DT_INT8 && !tilingIns_->bType_.isTrans);
        }
    } else {
        if (isA) {
            return tilingIns_->aType_.isTrans;
        } else {
            return !tilingIns_->bType_.isTrans;
        }
    }
}

void MatmulTilingAlgorithm::GetABL1Const(int32_t& aL1Const, int32_t& bL1Const, const L1StatusPack& l1Status) const
{
    int32_t reduceSize = static_cast<int32_t>(C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);

    aL1Const = C0_SIZE * C0_BYTE_SIZE * l1Status.dbAL1;
    if (tilingIns_->madType_ != MatrixMadType::MXMODE) {
        if (IsNeedAlign(true)) {
            aL1Const *= NUM_TWO;
        }
    } else {
        aL1Const = C0_SIZE * reduceSize * l1Status.dbAL1;
        if (tilingIns_->aType_.isTrans) {
            aL1Const *= (reduceSize / C0_SIZE);
        }
    }

    // 5/8 means 1/2(B Matrix size) + 1/8(Index Matrix size)
    bL1Const = tilingIns_->isSparse_ ? C0_SIZE * (C0_BYTE_SIZE / 8) * 5 * l1Status.dbBL1 :
        C0_SIZE * C0_BYTE_SIZE * l1Status.dbBL1;
    if (tilingIns_->madType_ == MatrixMadType::MXMODE) {
        bL1Const = C0_SIZE * reduceSize * l1Status.dbBL1;
        if (IsNeedAlign(false)) {
            bL1Const *= (reduceSize / C0_SIZE);
        }
    } else {
        if (IsNeedAlign(false)) {
            bL1Const *= NUM_TWO;
        }
    }
}

int32_t MatmulTilingAlgorithm::GetL1Size(const L1StatusPack& l1Status, const L0StatusPack& l0Status) const
{
    int32_t aL1Const = 0;
    int32_t bL1Const = 0;
    GetABL1Const(aL1Const, bL1Const, l1Status);

    const int32_t channelWiseL1Const = l1Status.channelWiseTimes * C0_SIZE * l1Status.dbBL1 * l0Status.dtypeBias;
    int32_t curAL1Size = 0;
    int32_t curBL1Size = 0;
    int32_t channelWiseL1Size = 0;
    int32_t dequantSize = 0;

    int32_t kaAlignValue = 1;
    int32_t kbAlignValue = 1;
    GetABL1KAlignValue(kaAlignValue, kbAlignValue);
    if (!MathUtil::CheckMulOverflow(l1Status.mAL1, l0Status.mL0, curAL1Size) ||
        !MathUtil::CheckMulOverflow(curAL1Size, aL1Const, curAL1Size) ||
        !MathUtil::CheckMulOverflow(curAL1Size, MathUtil::Align(l1Status.kAL1, kaAlignValue), curAL1Size)) {
        return 0;
    }
    if (!MathUtil::CheckMulOverflow(l1Status.nBL1, l0Status.nL0, curBL1Size) ||
        !MathUtil::CheckMulOverflow(curBL1Size, bL1Const, curBL1Size) ||
        !MathUtil::CheckMulOverflow(curBL1Size, MathUtil::Align(l1Status.kBL1, kbAlignValue), curBL1Size)) {
        return 0;
    }

    if (l1Status.channelWiseTimes > 0) {
        if (!MathUtil::CheckMulOverflow(l1Status.nBL1, l0Status.nL0, channelWiseL1Size) ||
            !MathUtil::CheckMulOverflow(channelWiseL1Size, channelWiseL1Const, channelWiseL1Size)) {
            return 0;
        }
    }

    if (tilingIns_->deqType == DequantType::TENSOR) {
        dequantSize = l1Status.nBL1 * l0Status.nL0 * C0_SIZE * UINT64_TYPES;
    }

    if (tilingIns_->aType_.pos == TPosition::TSCM) {
        curAL1Size = 0;
    }
    if (tilingIns_->bType_.pos == TPosition::TSCM) {
        curBL1Size = 0;
    }
    if (tilingIns_->biasType_.pos == TPosition::TSCM) {
        channelWiseL1Size = 0;
    }

    const int64_t totalSize = static_cast<int64_t>(curAL1Size) + static_cast<int64_t>(curBL1Size) +
        static_cast<int64_t>(channelWiseL1Size) + static_cast<int64_t>(dequantSize);
    return totalSize > INT_MAX ? INT_MAX : static_cast<int32_t>(totalSize);
}

int32_t MatmulTilingAlgorithm::CalL1MaxLen(int32_t resL1Size, L1StatusPack& l1Status, const L0StatusPack& l0Status,
    const int32_t alignValue, const L1TilingType axisName) const
{
    int32_t axisMaxLen = 1;
    if (axisName == L1TilingType::KAL1_16) {
        axisMaxLen = resL1Size /
            (l1Status.mAL1 * l0Status.mL0 * l1Status.dbAL1 * C0_SIZE * C0_BYTE_SIZE);
    }
    if (axisName == L1TilingType::KBL1_16) {
        axisMaxLen = resL1Size /
            (l1Status.nBL1 * l0Status.nL0 * l1Status.dbBL1 * C0_SIZE * C0_BYTE_SIZE);
    }
    axisMaxLen = MathUtil::AlignDown(axisMaxLen, alignValue);
    if (axisName == L1TilingType::M_AL1) {
        axisMaxLen = resL1Size / (MathUtil::Align(l1Status.kAL1, alignValue) * l0Status.mL0 * l1Status.dbAL1 * C0_SIZE *
            C0_BYTE_SIZE);
    }
    if (axisName == L1TilingType::N_BL1) {
        axisMaxLen = resL1Size / (MathUtil::Align(l1Status.kBL1, alignValue) * l0Status.nL0 * l1Status.dbBL1 * C0_SIZE *
            C0_BYTE_SIZE + l1Status.channelWiseTimes * l0Status.nL0 * C0_SIZE * C0_BYTE_SIZE);
    }
    return axisMaxLen;
}

/*
  brief:
    if factor greater then base, then factor = base
    if factor less than base, then get thr max factor of base, i.e. base 10, factor 9, then res factor = 5
*/
void MatmulTilingAlgorithm::GetNearestFactor(const int32_t& base, int32_t& factor, int32_t capValue) const
{
    if (!g_tempCfg.factorSplit) {
        return;
    }
    if (capValue == INT32_MAX) {
        capValue = base;
    }
    while ((factor > capValue) || (factor > 0 && base % factor != 0)) {
        factor--;
    }
}

void MatmulTilingAlgorithm::L1StatusAl1FullLoad(const CoreStatusPack& coreStatus, const L0StatusPack& l0Status,
    L1StatusPack& l1Status, int32_t res[][IDX_SEVEN]) const
{
    // if b matrix in L1, then b matrix must full load, goto b matrix full load patch
    if (tilingIns_->bType_.pos == TPosition::TSCM) {
        return;
    }
    const int32_t mRepeat = MathUtil::CeilDivision(coreStatus.m, l0Status.mL0);
    const int32_t nRepeat = MathUtil::CeilDivision(coreStatus.n, l0Status.nL0);
    int32_t kaAlignValue = 1;
    int32_t kbAlignValue = 1;
    GetABL1KAlignValue(kaAlignValue, kbAlignValue);
    l1Status.kAL1 = MathUtil::CeilDivision(l1Status.kAL1, l0Status.kL0) * l0Status.kL0;
    const int32_t curL1Size = GetL1Size(l1Status, l0Status);
    const int32_t a1Length = GetAL1UbSize(l1Status, l0Status);
    if (curL1Size > 0 && curL1Size <= tilingIns_->bufferPool_.l1Size && a1Length < tilingIns_->bufferPool_.ubSize) {
        l1Status.aL1FullLoad = true;
        l1Status.aL1Size =
            max(MathUtil::Align(coreStatus.k, kaAlignValue), MathUtil::Align(l1Status.kAL1, kaAlignValue)) *
            max(l1Status.mAL1 * l0Status.mL0, coreStatus.m) * C0_SIZE * C0_BYTE_SIZE;
        if (tilingIns_->aType_.pos == TPosition::TSCM) {
            l1Status.bL1Size = tilingIns_->bufferPool_.l1Size;
        } else {
            l1Status.bL1Size = tilingIns_->bufferPool_.l1Size - l1Status.aL1Size;
        }
        if (g_tempCfg.l1DB == DB_ON) {
            l1Status.dbBL1 = DB_ON;
            if (GetL1Size(l1Status, l0Status) > tilingIns_->bufferPool_.l1Size) {
                l1Status.dbBL1 = DB_OFF;
            }
        }
        const int32_t biasSize =
            l1Status.channelWiseTimes * l1Status.nBL1 * l0Status.nL0 * C0_SIZE * l0Status.dtypeBias * l1Status.dbBL1;
        int32_t dequantSize = 0;
        if (tilingIns_->deqType == DequantType::TENSOR) {
            dequantSize = l1Status.nBL1 * l0Status.nL0 * C0_SIZE * UINT64_TYPES;
        }
        l1Status.kBL1 = min(CalL1MaxLen((l1Status.bL1Size - biasSize - dequantSize), l1Status, l0Status, kbAlignValue,
            L1TilingType::KBL1_16),
            coreStatus.k);
        if (IsUbNd2Nz()) {
            l1Status.dbBL1 = DB_OFF;
            const int32_t b1Length = tilingIns_->bufferPool_.ubSize - a1Length;
            l1Status.kBL1 = min(CalL1MaxLen(min(l1Status.bL1Size - biasSize - dequantSize, b1Length),
                l1Status, l0Status, kbAlignValue, L1TilingType::KBL1_16), coreStatus.k);
        }
        l1Status.bL1Times = min(l1Status.kBL1 / l0Status.kL0, l1Status.maxKBL1);
        GetNearestFactor(l1Status.allTimes, l1Status.bL1Times); // tik-mm support no factor---ncheck
        l1Status.kBL1 = l1Status.bL1Times * l0Status.kL0;
        if (l1Status.kBL1 == coreStatus.k) {
            l1Status.nBL1 = min(CalL1MaxLen(l1Status.bL1Size, l1Status, l0Status, kbAlignValue, L1TilingType::N_BL1),
                l1Status.maxNBL1);
            GetNearestFactor(nRepeat, l1Status.nBL1);
        }

        const bool invalidL1Status = (l1Status.nBL1 == 0 || l1Status.kBL1 == 0) ? true : false;
        const int32_t possibleMRepeat = (l1Status.kBL1 == coreStatus.k) ? 1 : mRepeat;
        // m+n*m_repeat XXX---ncheck
        l1Status.loadSize = invalidL1Status ?
            INT32_MAX :
            ((tilingIns_->aType_.pos == TPosition::TSCM ? 0 : coreStatus.m) + possibleMRepeat * coreStatus.n);
        if (g_tempCfg.l1DB == DB_ON && l1Status.kBL1 == coreStatus.k && l1Status.nBL1 * l0Status.nL0 == coreStatus.n) {
            l1Status.dbBL1 = DB_OFF;
        }
        res[IDX_ONE][IDX_ZERO] = l1Status.kAL1;
        res[IDX_ONE][IDX_ONE] = l1Status.mAL1;
        res[IDX_ONE][IDX_TWO] = l1Status.dbAL1;
        res[IDX_ONE][IDX_THREE] = l1Status.kBL1;
        res[IDX_ONE][IDX_FOUR] = l1Status.nBL1;
        res[IDX_ONE][IDX_FIVE] = l1Status.dbBL1;
        res[IDX_ONE][IDX_SIX] = l1Status.loadSize;
    }
}

void MatmulTilingAlgorithm::L1StatusBl1FullLoad(const CoreStatusPack& coreStatus, const L0StatusPack& l0Status,
    L1StatusPack& l1Status, int32_t res[][IDX_SEVEN]) const
{
    // if a matrix in L1, then a matrix must full load, goto a matrix full load patch
    if (tilingIns_->aType_.pos == TPosition::TSCM) {
        return;
    }
    const int32_t mRepeat = MathUtil::CeilDivision(coreStatus.m, l0Status.mL0);
    const int32_t nRepeat = MathUtil::CeilDivision(coreStatus.n, l0Status.nL0);
    int32_t kaAlignValue = 1;
    int32_t kbAlignValue = 1;
    GetABL1KAlignValue(kaAlignValue, kbAlignValue);
    l1Status.kBL1 = MathUtil::CeilDivision(l1Status.kBL1, l0Status.kL0) * l0Status.kL0;
    const int32_t curL1Size = GetL1Size(l1Status, l0Status);
    const int32_t b1Length = GetBL1UbSize(l1Status, l0Status);
    if (curL1Size > 0 && curL1Size <= tilingIns_->bufferPool_.l1Size && b1Length < tilingIns_->bufferPool_.ubSize) {
        l1Status.bL1FullLoad = true;
        l1Status.bL1Size =
            max(MathUtil::Align(coreStatus.k, kbAlignValue), MathUtil::Align(l1Status.kBL1, kbAlignValue)) *
            max(l1Status.nBL1 * l0Status.nL0, coreStatus.n) * C0_SIZE * C0_BYTE_SIZE;
        if (tilingIns_->bType_.pos == TPosition::TSCM) {
            l1Status.aL1Size = tilingIns_->bufferPool_.l1Size;
        } else {
            l1Status.aL1Size = tilingIns_->bufferPool_.l1Size - l1Status.bL1Size;
        }
        if (g_tempCfg.l1DB == DB_ON) {
            l1Status.dbAL1 = DB_ON;
            if (GetL1Size(l1Status, l0Status) > tilingIns_->bufferPool_.l1Size) {
                l1Status.dbAL1 = DB_OFF;
            }
        }
        int32_t dequantSize = 0;
        if (tilingIns_->deqType == DequantType::TENSOR) {
            dequantSize = l1Status.nBL1 * l0Status.nL0 * C0_SIZE * UINT64_TYPES;
        }
        const int32_t biasSize =
            l1Status.channelWiseTimes * l1Status.nBL1 * l0Status.nL0 * C0_SIZE * l0Status.dtypeBias * l1Status.dbBL1;
        l1Status.kAL1 = min(CalL1MaxLen((l1Status.aL1Size - biasSize - dequantSize), l1Status, l0Status, kaAlignValue,
            L1TilingType::KAL1_16),
            coreStatus.k);
        if (IsUbNd2Nz()) {
            l1Status.dbAL1 = DB_OFF;
            const int32_t a1Length = tilingIns_->bufferPool_.ubSize - b1Length;
            l1Status.kAL1 = min(CalL1MaxLen(min(l1Status.aL1Size - biasSize - dequantSize, a1Length),
                l1Status, l0Status, kaAlignValue, L1TilingType::KAL1_16), coreStatus.k);
        }
        l1Status.aL1Times = min(l1Status.kAL1 / l0Status.kL0, l1Status.maxKAL1);
        GetNearestFactor(l1Status.allTimes, l1Status.aL1Times); // tik-mm support no factor---ncheck
        l1Status.kAL1 = l1Status.aL1Times * l0Status.kL0;
        if (l1Status.kAL1 == coreStatus.k) {
            l1Status.mAL1 =
                min(CalL1MaxLen(l1Status.aL1Size - biasSize, l1Status, l0Status, kaAlignValue, L1TilingType::M_AL1),
                l1Status.maxMAL1);
            GetNearestFactor(mRepeat, l1Status.mAL1); // tik-mm support no factor---ncheck
        }

        const bool invalidL1Status = (l1Status.mAL1 == 0 || l1Status.kAL1 == 0) ? true : false;
        const int32_t possibleNRepeat = (l1Status.kAL1 == coreStatus.k) ? 1 : nRepeat; // no repeat---ncheck
        l1Status.loadSize = invalidL1Status ?
            INT32_MAX :
            ((tilingIns_->bType_.pos == TPosition::TSCM ? 0 : coreStatus.n) + possibleNRepeat * coreStatus.m);
        if (g_tempCfg.l1DB == DB_ON && l1Status.kAL1 == coreStatus.k && l1Status.mAL1 * l0Status.mL0 >= coreStatus.m) {
            l1Status.dbAL1 = DB_OFF;
        }
        res[IDX_TWO][IDX_ZERO] = l1Status.kAL1;
        res[IDX_TWO][IDX_ONE] = l1Status.mAL1;
        res[IDX_TWO][IDX_TWO] = l1Status.dbAL1;
        res[IDX_TWO][IDX_THREE] = l1Status.kBL1;
        res[IDX_TWO][IDX_FOUR] = l1Status.nBL1;
        res[IDX_TWO][IDX_FIVE] = l1Status.dbBL1;
        res[IDX_TWO][IDX_SIX] = l1Status.loadSize;
    }
}

void MatmulTilingAlgorithm::L1StatusBothFullLoad(const CoreStatusPack& coreStatus, const L0StatusPack& l0Status,
    L1StatusPack& l1Status, int32_t res[][IDX_SEVEN]) const
{
    l1Status.kAL1 = MathUtil::CeilDivision(l1Status.kAL1, l0Status.kL0) * l0Status.kL0;
    l1Status.kBL1 = MathUtil::CeilDivision(l1Status.kBL1, l0Status.kL0) * l0Status.kL0;
    const int32_t curL1Size = GetL1Size(l1Status, l0Status);
    const int32_t a1Length = GetAL1UbSize(l1Status, l0Status);
    const int32_t b1Length = GetBL1UbSize(l1Status, l0Status);
    if (((curL1Size > 0 && curL1Size <= tilingIns_->bufferPool_.l1Size) &&
        (a1Length + b1Length) <= tilingIns_->bufferPool_.ubSize) ||
        (tilingIns_->aType_.pos == TPosition::TSCM && tilingIns_->bType_.pos == TPosition::TSCM)) {
        l1Status.bothFullLoad = true;
        l1Status.loadSize = (tilingIns_->aType_.pos == TPosition::TSCM ? 0 : coreStatus.m) +
            (tilingIns_->bType_.pos == TPosition::TSCM ? 0 : coreStatus.n);
        res[IDX_ZERO][IDX_ZERO] = l1Status.kAL1;
        res[IDX_ZERO][IDX_ONE] = l1Status.mAL1;
        res[IDX_ZERO][IDX_TWO] = l1Status.dbAL1;
        res[IDX_ZERO][IDX_THREE] = l1Status.kBL1;
        res[IDX_ZERO][IDX_FOUR] = l1Status.nBL1;
        res[IDX_ZERO][IDX_FIVE] = l1Status.dbBL1;
        res[IDX_ZERO][IDX_SIX] = l1Status.loadSize;
    }
}
void MatmulTilingAlgorithm::NeitherFullLoadDb(const CoreStatusPack& coreStatus, const L0StatusPack& l0Status,
    L1StatusPack& l1Status, const int32_t& kbl1Db) const
{
    const int32_t tmpKbl116 = l1Status.kBL1;
    l1Status.kBL1 = kbl1Db;
    if (GetL1Size(l1Status, l0Status) > tilingIns_->bufferPool_.l1Size ||
        (GetAL1UbSize(l1Status, l0Status) + GetBL1UbSize(l1Status, l0Status)) > tilingIns_->bufferPool_.ubSize) {
        l1Status.dbBL1 = DB_OFF;
        if (GetL1Size(l1Status, l0Status) > tilingIns_->bufferPool_.l1Size ||
            GetAL1UbSize(l1Status, l0Status) + GetBL1UbSize(l1Status, l0Status) > tilingIns_->bufferPool_.ubSize) {
            l1Status.dbAL1 = DB_OFF;
        }
    }
    l1Status.kBL1 = coreStatus.k;
    const bool bothDoubleBuffer = coreStatus.m != l0Status.mL0 && coreStatus.k > l0Status.kL0 &&
        (GetL1Size(l1Status, l0Status) > tilingIns_->bufferPool_.l1Size ||
         (GetAL1UbSize(l1Status, l0Status) + GetBL1UbSize(l1Status, l0Status)) > tilingIns_->bufferPool_.ubSize);
    l1Status.kBL1 = tmpKbl116;
    if (bothDoubleBuffer) {
        l1Status.dbAL1 = DB_ON;
        l1Status.dbBL1 = DB_ON;
        if (GetL1Size(l1Status, l0Status) > tilingIns_->bufferPool_.l1Size ||
            (GetAL1UbSize(l1Status, l0Status) + GetBL1UbSize(l1Status, l0Status)) > tilingIns_->bufferPool_.ubSize) {
            l1Status.dbBL1 = DB_OFF;
            if (GetL1Size(l1Status, l0Status) > tilingIns_->bufferPool_.l1Size ||
                GetAL1UbSize(l1Status, l0Status) + GetBL1UbSize(l1Status, l0Status) > tilingIns_->bufferPool_.ubSize) {
                l1Status.dbAL1 = DB_OFF;
            }
        }
    }
}

void MatmulTilingAlgorithm::NeitherFullLoadMN(const CoreStatusPack& coreStatus, const L0StatusPack& l0Status,
    L1StatusPack& l1Status) const
{
    const int32_t mRepeat = MathUtil::CeilDivision(coreStatus.m, l0Status.mL0);
    int32_t nRepeat = MathUtil::CeilDivision(coreStatus.n, l0Status.nL0);
    if (l0Status.dtypeBias == FP32_BYTES && l1Status.channelWiseTimes > 0) {
        l1Status.channelWiseTimes++;
    }
    int32_t biasSize = l1Status.channelWiseTimes * l1Status.nBL1 * l0Status.nL0 * C0_SIZE * FP16_BYTES * l1Status.dbBL1;
    int32_t dequantSize = 0;
    if (tilingIns_->deqType == DequantType::TENSOR) {
        dequantSize = l1Status.nBL1 * l0Status.nL0 * C0_SIZE * UINT64_TYPES;
    }
    // Align value is used in FP32 in FP32 out data flow mode
    int32_t kaAlignValue = 1;
    int32_t kbAlignValue = 1;
    GetABL1KAlignValue(kaAlignValue, kbAlignValue);
    L1StatusPack l1Mfirst;
    L1StatusPack l1Nfirst;
    errno_t err =
        memcpy_s(static_cast<void*>(&l1Mfirst), sizeof(l1Mfirst), static_cast<void*>(&l1Status), sizeof(l1Mfirst));
    if (err != EOK) {
        TILING_LOG_ERROR("memcpy error");
        return;
    }
    err = memcpy_s(static_cast<void*>(&l1Nfirst), sizeof(l1Nfirst), static_cast<void*>(&l1Status), sizeof(l1Nfirst));
    if (err != EOK) {
        TILING_LOG_ERROR("memcpy error");
    }
    // default l1Status.nBL1 = 1
    // calculate M first condition
    l1Mfirst.bL1Size = MathUtil::Align(l1Mfirst.kBL1, kbAlignValue) * l0Status.nL0 * C0_SIZE * C0_BYTE_SIZE *
        l1Mfirst.dbBL1;
    l1Mfirst.aL1Size = tilingIns_->bufferPool_.l1Size - l1Mfirst.bL1Size;
    int32_t a1Length = tilingIns_->bufferPool_.ubSize - GetBL1UbSize(l1Mfirst, l0Status);
    l1Mfirst.mAL1 = max(min(min(
        CalL1MaxLen(l1Mfirst.aL1Size - biasSize - dequantSize, l1Mfirst, l0Status, kaAlignValue, L1TilingType::M_AL1),
        l1Mfirst.maxMAL1),
        mRepeat),
        1);
    if (IsUbNd2Nz()) {
        l1Mfirst.mAL1 = max(min(min(
            CalL1MaxLen(min(l1Mfirst.aL1Size - biasSize - dequantSize, a1Length), l1Mfirst, l0Status, kaAlignValue, L1TilingType::M_AL1),
            l1Mfirst.maxMAL1),
            mRepeat),
            1);
    }
    GetNearestFactor(mRepeat, l1Mfirst.mAL1); // tik-mm support no factor ----ncheck
    l1Mfirst.aL1Size = MathUtil::Align(l1Mfirst.kAL1, kaAlignValue) * l1Mfirst.mAL1 * l0Status.mL0 * C0_SIZE *
        C0_BYTE_SIZE * l1Mfirst.dbAL1;
    l1Mfirst.bL1Size = tilingIns_->bufferPool_.l1Size - l1Mfirst.aL1Size;
    int32_t b1Length = tilingIns_->bufferPool_.ubSize - GetAL1UbSize(l1Mfirst, l0Status);
    l1Mfirst.nBL1 = max(min(min(
        CalL1MaxLen(l1Mfirst.bL1Size - biasSize - dequantSize, l1Mfirst, l0Status, kbAlignValue, L1TilingType::N_BL1),
        l1Mfirst.maxNBL1),
        nRepeat),
        1);
    if (IsUbNd2Nz()) {
        l1Mfirst.nBL1 = max(min(min(
            CalL1MaxLen(min(l1Mfirst.bL1Size - biasSize - dequantSize, b1Length), l1Mfirst, l0Status, kbAlignValue, L1TilingType::N_BL1),
            l1Mfirst.maxNBL1),
            nRepeat),
            1);
    }
    GetNearestFactor(nRepeat, l1Mfirst.nBL1);
    l1Mfirst.loadSize =
        coreStatus.m + coreStatus.n * MathUtil::CeilDivision(coreStatus.m, l1Mfirst.mAL1 * l0Status.mL0);

    // calculate N first condition
    l1Nfirst.aL1Size = MathUtil::Align(l1Nfirst.kAL1, kaAlignValue) * l0Status.mL0 * C0_SIZE * C0_BYTE_SIZE *
        l1Nfirst.dbAL1;
    l1Nfirst.bL1Size = tilingIns_->bufferPool_.l1Size - l1Nfirst.aL1Size;
    b1Length = tilingIns_->bufferPool_.ubSize - GetAL1UbSize(l1Nfirst, l0Status);
    l1Nfirst.nBL1 = max(min(min(
        CalL1MaxLen(l1Nfirst.bL1Size - biasSize - dequantSize, l1Nfirst, l0Status, kbAlignValue, L1TilingType::N_BL1),
        l1Nfirst.maxNBL1),
        nRepeat),
        1);
    if (IsUbNd2Nz()) {
        l1Nfirst.nBL1 = max(min(min(
            CalL1MaxLen(min(l1Nfirst.bL1Size - biasSize - dequantSize, b1Length), l1Nfirst, l0Status, kbAlignValue, L1TilingType::N_BL1),
            l1Nfirst.maxNBL1),
            nRepeat),
            1);
    }
    GetNearestFactor(nRepeat, l1Nfirst.nBL1);
    l1Nfirst.bL1Size = MathUtil::Align(coreStatus.k, kbAlignValue) * l1Nfirst.nBL1 * l0Status.nL0 * C0_SIZE *
        C0_BYTE_SIZE * l1Nfirst.dbBL1;
    l1Nfirst.aL1Size = tilingIns_->bufferPool_.l1Size - l1Nfirst.bL1Size;
    a1Length = tilingIns_->bufferPool_.ubSize - GetBL1UbSize(l1Nfirst, l0Status);
    biasSize = biasSize * l1Nfirst.nBL1;
    l1Nfirst.mAL1 = max(min(min(
        CalL1MaxLen(l1Nfirst.aL1Size - biasSize - dequantSize, l1Nfirst, l0Status, kaAlignValue, L1TilingType::M_AL1),
        l1Nfirst.maxMAL1),
        mRepeat),
        1);
    if (IsUbNd2Nz()) {
        l1Nfirst.mAL1 = max(min(min(
            CalL1MaxLen(min(l1Nfirst.aL1Size - biasSize - dequantSize, a1Length), l1Nfirst, l0Status, kaAlignValue, L1TilingType::M_AL1),
            l1Nfirst.maxMAL1),
            mRepeat),
            1);
    }
    GetNearestFactor(mRepeat, l1Nfirst.mAL1);
    l1Nfirst.loadSize =
        coreStatus.m * MathUtil::CeilDivision(coreStatus.n, l1Nfirst.nBL1 * l0Status.nL0) + coreStatus.n;

    if (l1Status.kAL1 >= coreStatus.k && l1Status.kBL1 >= coreStatus.k) {
        if (l1Nfirst.loadSize > l1Mfirst.loadSize) {
            const errno_t errnoT = memcpy_s(&l1Status, sizeof(l1Status), &l1Mfirst, sizeof(l1Status));
            if (errnoT != EOK) {
                TILING_LOG_ERROR("memcpy error");
                return;
            }
        } else {
            const errno_t errnoT = memcpy_s(&l1Status, sizeof(l1Status), &l1Nfirst, sizeof(l1Status));
            if (errnoT != EOK) {
                TILING_LOG_ERROR("memcpy error");
                return;
            }
        }
    }
    if (l1Status.kAL1 >= coreStatus.k && l1Status.kBL1 < coreStatus.k) {
        l1Mfirst.nBL1 = 1;
        const errno_t errnoT = memcpy_s(&l1Status, sizeof(l1Status), &l1Mfirst, sizeof(l1Status));
        if (errnoT != EOK) {
            TILING_LOG_ERROR("memcpy error");
            return;
        }
    }
    if (l1Status.kAL1 < coreStatus.k && l1Status.kBL1 >= coreStatus.k) {
        l1Nfirst.mAL1 = 1;
        const errno_t errnoT = memcpy_s(&l1Status, sizeof(l1Status), &l1Nfirst, sizeof(l1Status));
        if (errnoT != EOK) {
            TILING_LOG_ERROR("memcpy error");
            return;
        }
    }
    if (l1Status.kAL1 < coreStatus.k && l1Status.kBL1 < coreStatus.k) {
        l1Status.mAL1 = 1;
        l1Status.nBL1 = 1;
        l1Status.loadSize = coreStatus.m * MathUtil::CeilDivision(coreStatus.n, l1Mfirst.nBL1 * l0Status.nL0) +
            coreStatus.n * MathUtil::CeilDivision(coreStatus.m, l1Mfirst.mAL1 * l0Status.mL0);
    }
}

void MatmulTilingAlgorithm::NeitherFullLoadKforNZ(const CoreStatusPack& coreStatus, const L0StatusPack& l0Status,
    L1StatusPack& l1Status) const
{
    l1Status.kBL1 = coreStatus.k;
    const int32_t biasSize =
        l1Status.channelWiseTimes * l1Status.nBL1 * l0Status.nL0 * C0_SIZE * l0Status.dtypeBias * l1Status.dbBL1;
    int32_t dequantSize = 0;
    if (tilingIns_->deqType == DequantType::TENSOR) {
        dequantSize = l1Status.nBL1 * l0Status.nL0 * C0_SIZE * UINT64_TYPES;
    }
    int32_t kaAlignValue = 1;
    int32_t kbAlignValue = 1;
    GetABL1KAlignValue(kaAlignValue, kbAlignValue);

    if (GetL1Size(l1Status, l0Status) > 0 && GetL1Size(l1Status, l0Status) <= tilingIns_->bufferPool_.l1Size) {
        l1Status.bL1Size = MathUtil::Align(coreStatus.k, kbAlignValue) * l1Status.nBL1 * l0Status.nL0 * C0_SIZE *
            C0_BYTE_SIZE * l1Status.dbBL1;
        l1Status.aL1Size = tilingIns_->bufferPool_.l1Size - l1Status.bL1Size;
        int32_t a1Length = tilingIns_->bufferPool_.ubSize - GetBL1UbSize(l1Status, l0Status);
        l1Status.kAL1 = min(CalL1MaxLen(l1Status.aL1Size - biasSize - dequantSize, l1Status, l0Status, kaAlignValue,
            L1TilingType::KAL1_16),
            coreStatus.k);
        if (IsUbNd2Nz()) {
            l1Status.kAL1 = min(CalL1MaxLen(min(l1Status.aL1Size - biasSize - dequantSize, a1Length), l1Status, l0Status, kaAlignValue,
                L1TilingType::KAL1_16),
                coreStatus.k);
        }
        
        l1Status.aL1Times = max(min(l1Status.kAL1 / l0Status.kL0, l1Status.maxKAL1), 1);
        GetNearestFactor(l1Status.allTimes, l1Status.aL1Times);
        l1Status.kAL1 = l1Status.aL1Times * l0Status.kL0;
    } else {
        // when NeitherFullLoadMN change the nBL1 and mAL1
        int32_t perK = min((tilingIns_->bufferPool_.l1Size - biasSize - dequantSize) /
            (l0Status.mL0 * C0_SIZE * C0_BYTE_SIZE * l1Status.dbAL1 +
            C0_SIZE * l0Status.nL0 * C0_BYTE_SIZE * l1Status.dbBL1) /
            l0Status.kL0 * l0Status.kL0,
            coreStatus.k);
        if (IsUbNd2Nz()) {
            perK = min(min(tilingIns_->bufferPool_.l1Size - biasSize - dequantSize,
                tilingIns_->bufferPool_.ubSize) /
                (l0Status.mL0 * C0_SIZE * C0_BYTE_SIZE * l1Status.dbAL1 +
                C0_SIZE * l0Status.nL0 * C0_BYTE_SIZE * l1Status.dbBL1) /
                l0Status.kL0 * l0Status.kL0,
                coreStatus.k);
        }
        const int32_t biasFactor = tilingIns_->isBias ? l1Status.nBL1 * l0Status.nL0 : 0;
        const int32_t aAlignedPerK = MathUtil::Align(perK, kaAlignValue);
        const int32_t bAlignedPerK = MathUtil::Align(perK, kbAlignValue);
        if (tilingIns_->aType_.dataType == DataType::DT_FLOAT &&
            !CheckL1Size(l1Status.mAL1 * l0Status.mL0 * aAlignedPerK * l1Status.dbAL1,
            l1Status.nBL1 * l0Status.nL0 * bAlignedPerK * l1Status.dbBL1,
            biasFactor * C0_SIZE * l0Status.dtypeBias * l1Status.dbBL1 + dequantSize)) {
            perK -= 1;
        }
        int32_t perTimes = min(perK / l0Status.kL0, max(l1Status.maxKAL1, l1Status.maxKBL1));
        GetNearestFactor(l1Status.allTimes, perTimes);
        perTimes = min(perTimes, l1Status.allTimes);
        perK = perTimes * l0Status.kL0;
        l1Status.kAL1 = perK;
        l1Status.kBL1 = perK;
    }
}

bool MatmulTilingAlgorithm::CheckL1Size(int32_t amat, int32_t bmat, int32_t curBiasL1Size) const
{
    const int64_t loadSizeBytes = (static_cast<int64_t>(amat + bmat) * C0_SIZE * C0_BYTE_SIZE +
        static_cast<int64_t>(curBiasL1Size));
    return loadSizeBytes <= tilingIns_->bufferPool_.l1Size;
}

void MatmulTilingAlgorithm::NeitherFullLoadKforND(const CoreStatusPack& coreStatus, const L0StatusPack& l0Status,
    L1StatusPack& l1Status, const int32_t& kMaxAxis) const
{
    int32_t biasSize =
        l1Status.channelWiseTimes * l1Status.nBL1 * l0Status.nL0 * C0_SIZE * l0Status.dtypeBias * l1Status.dbBL1;
    int32_t dequantSize = 0;
    if (tilingIns_->deqType == DequantType::TENSOR) {
        dequantSize = l1Status.nBL1 * l0Status.nL0 * C0_SIZE * UINT64_TYPES;
    }
    int32_t alignValue = FP32_ALIGN_SIZE;
    if (DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8)) {
        alignValue = INT8_ALIGN_SIZE;
    } else if (DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT4)) {
        alignValue = INT4_ALIGN_SIZE;
    }
    const int32_t reduceSize = C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE;
    const int32_t alignM = MathUtil::CeilDivision(l1Status.mAL1 * C0_SIZE, alignValue) * alignValue;
    const int32_t alignN = MathUtil::CeilDivision(l1Status.nBL1 * C0_SIZE, alignValue) * alignValue;
    const int32_t alignK = MathUtil::CeilDivision(l0Status.kL0 * reduceSize, alignValue) * alignValue *
        DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
    if (kMaxAxis == 1) {
        // first get k_al1, second get k_bl1
        l1Status.kBL1 = l0Status.kL0;
        if ((tilingIns_->bType_.dataType == DataType::DT_FLOAT) ||
            (tilingIns_->aType_.isTrans && DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8))) {
            l1Status.bL1Size = l1Status.kBL1 * l0Status.nL0 * C0_SIZE * alignK * l1Status.nBL1 * l1Status.dbBL1;
        } else if (!tilingIns_->bType_.isTrans && (DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8)
            || DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT4))) {
            l1Status.bL1Size = l1Status.kBL1 * l0Status.nL0 * alignK * alignN * l1Status.dbBL1;
        } else {
            l1Status.bL1Size = l1Status.kBL1 * l1Status.nBL1 * l0Status.nL0 * C0_SIZE * C0_BYTE_SIZE * l1Status.dbBL1;
        }
        l1Status.aL1Size = tilingIns_->bufferPool_.l1Size - l1Status.bL1Size;
        int32_t a1Length = tilingIns_->bufferPool_.ubSize - GetBL1UbSize(l1Status, l0Status);
        auto factor = l1Status.mAL1 * l0Status.mL0 * C0_SIZE * l1Status.dbAL1 * C0_BYTE_SIZE;
        l1Status.kAL1 = (factor == 0) ? coreStatus.k : min((l1Status.aL1Size - biasSize - dequantSize) / factor,
                coreStatus.k);
        if (IsUbNd2Nz()) {
            l1Status.kAL1 = (factor == 0) ? coreStatus.k : min(min(l1Status.aL1Size - biasSize - dequantSize, a1Length) / factor,
                coreStatus.k);
        }
        
        l1Status.aL1Times = max(l1Status.kAL1 / l0Status.kL0, 1);
        GetNearestFactor(l1Status.allTimes, l1Status.aL1Times); // tik-mm support no factor ----ncheck
        l1Status.kAL1 = l1Status.aL1Times * l0Status.kL0;
        l1Status.aL1Size = l1Status.kAL1 * l1Status.mAL1 * l0Status.mL0 * C0_SIZE * C0_BYTE_SIZE * l1Status.dbAL1;
        l1Status.bL1Size = tilingIns_->bufferPool_.l1Size - l1Status.aL1Size;
        int32_t b1Length = tilingIns_->bufferPool_.ubSize - GetAL1UbSize(l1Status, l0Status);
        if ((tilingIns_->bType_.dataType == DataType::DT_FLOAT) || (tilingIns_->aType_.isTrans &&
            DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8))) {
            l1Status.kBL1 = min((l1Status.bL1Size - biasSize - dequantSize) /
                    (l1Status.nBL1 * l0Status.nL0 * C0_SIZE * l1Status.dbBL1 * alignK),
                    coreStatus.k);
            if (IsUbNd2Nz()) {
                l1Status.kBL1 = min(min(l1Status.bL1Size - biasSize - dequantSize, b1Length) /
                    (l1Status.nBL1 * l0Status.nL0 * C0_SIZE * l1Status.dbBL1 * alignK),
                    coreStatus.k);
            }
        } else if (!tilingIns_->bType_.isTrans &&
            (DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8) ||
            DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT4))) {
            l1Status.kBL1 = min((l1Status.bL1Size - biasSize - dequantSize) /
                    (alignN * l0Status.nL0 * l1Status.dbBL1 * alignK),
                    coreStatus.k);
            if (IsUbNd2Nz()) {
                l1Status.kBL1 = min(min(l1Status.bL1Size - biasSize - dequantSize, b1Length) /
                    (alignN * l0Status.nL0 * l1Status.dbBL1 * alignK),
                    coreStatus.k);
            }            
        } else {
            l1Status.kBL1 = min((l1Status.bL1Size - biasSize - dequantSize)/
                (l1Status.nBL1 * l0Status.nL0 * C0_SIZE * l1Status.dbBL1 * C0_BYTE_SIZE),
                coreStatus.k);
            if (IsUbNd2Nz()) {
                l1Status.kBL1 = min(min(l1Status.bL1Size - biasSize - dequantSize, b1Length)/
                    (l1Status.nBL1 * l0Status.nL0 * C0_SIZE * l1Status.dbBL1 * C0_BYTE_SIZE),
                    coreStatus.k);}
        }
        l1Status.bL1Times = max(min(l1Status.kBL1 / l0Status.kL0, l1Status.maxKBL1), 1);
        GetNearestFactor(l1Status.allTimes, l1Status.bL1Times);
        l1Status.kBL1 = l1Status.bL1Times * l0Status.kL0;
    }
    if (kMaxAxis == NUM_TWO) {
        // first get k_bl1, second get k_al1
        l1Status.kAL1 = l0Status.kL0;
        if ((tilingIns_->aType_.isTrans && tilingIns_->aType_.dataType == DataType::DT_FLOAT) ||
            (!tilingIns_->aType_.isTrans && DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8))) {
            l1Status.aL1Size = l1Status.kAL1 * l1Status.mAL1 * l0Status.mL0 * C0_SIZE * alignK * l1Status.dbAL1;
        } else if (tilingIns_->aType_.isTrans &&
            DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8)) {
            l1Status.aL1Size = l1Status.kAL1 * alignM * l0Status.mL0 * alignK * l1Status.dbAL1;
        } else {
            l1Status.aL1Size = l1Status.kAL1 * l1Status.mAL1 * l0Status.mL0 * C0_SIZE * C0_BYTE_SIZE * l1Status.dbAL1;
        }

        l1Status.bL1Size = tilingIns_->bufferPool_.l1Size - l1Status.aL1Size;
        int32_t b1Length = tilingIns_->bufferPool_.ubSize - GetAL1UbSize(l1Status, l0Status);
        l1Status.kBL1 = min((l1Status.bL1Size - biasSize - dequantSize) /
            (l1Status.nBL1 * l0Status.nL0 * C0_SIZE * l1Status.dbBL1 * C0_BYTE_SIZE),
            coreStatus.k);
        if (IsUbNd2Nz()) {
            l1Status.kBL1 = min(min(l1Status.bL1Size - biasSize - dequantSize, b1Length) /
                (l1Status.nBL1 * l0Status.nL0 * C0_SIZE * l1Status.dbBL1 * C0_BYTE_SIZE),
                coreStatus.k);
        }
        l1Status.bL1Times = max(l1Status.kBL1 / l0Status.kL0, 1);
        GetNearestFactor(l1Status.allTimes, l1Status.bL1Times);
        l1Status.kBL1 = l1Status.bL1Times * l0Status.kL0;
        l1Status.bL1Size = l1Status.kBL1 * l1Status.nBL1 * l0Status.nL0 * C0_SIZE * C0_BYTE_SIZE * l1Status.dbBL1;
        l1Status.aL1Size = tilingIns_->bufferPool_.l1Size - l1Status.bL1Size;
        int32_t a1Length = tilingIns_->bufferPool_.ubSize - GetBL1UbSize(l1Status, l0Status);
        if ((tilingIns_->aType_.isTrans && tilingIns_->aType_.dataType == DataType::DT_FLOAT) ||
            (!tilingIns_->aType_.isTrans && (DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8) ||
            DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT4)))) {
            auto factor = l1Status.mAL1 * l0Status.mL0 * C0_SIZE * l1Status.dbAL1 * alignK;
            l1Status.kAL1 = (factor == 0) ? coreStatus.k : min((l1Status.aL1Size - biasSize - dequantSize) / factor,
                coreStatus.k);
            if (IsUbNd2Nz()) {
                l1Status.kAL1 = (factor == 0) ? coreStatus.k : min(min(l1Status.aL1Size - biasSize - dequantSize, a1Length) / factor,
                    coreStatus.k);
            }
        } else if (tilingIns_->aType_.isTrans && DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8)) {
            l1Status.kAL1 = min((l1Status.aL1Size - biasSize - dequantSize) /
                (alignM * l0Status.mL0 * l1Status.dbAL1 * alignK), coreStatus.k);
            if (IsUbNd2Nz()) {
                l1Status.kAL1 = min(min(l1Status.aL1Size - biasSize - dequantSize, a1Length) /
                    (alignM * l0Status.mL0 * l1Status.dbAL1 * alignK), coreStatus.k);
            }
            l1Status.aL1Size = l1Status.kAL1 * alignM * l0Status.mL0 * alignK * l1Status.dbAL1;
        } else {
            auto factor = l1Status.mAL1 * l0Status.mL0 * C0_SIZE * l1Status.dbAL1 * C0_BYTE_SIZE;
            l1Status.kAL1 = (factor == 0) ? coreStatus.k : min((l1Status.aL1Size - biasSize - dequantSize) / factor,
                coreStatus.k);
            if (IsUbNd2Nz()) {
                l1Status.kAL1 = (factor == 0) ? coreStatus.k : min(min(l1Status.aL1Size - biasSize - dequantSize, a1Length) / factor,
                    coreStatus.k);
            }
        }
        l1Status.aL1Times = max(min(l1Status.kAL1 / l0Status.kL0, l1Status.maxKAL1), 1);
        GetNearestFactor(l1Status.allTimes, l1Status.aL1Times);
        l1Status.kAL1 = l1Status.aL1Times * l0Status.kL0;
    }
}
void MatmulTilingAlgorithm::NeitherFullLoadK(const CoreStatusPack& coreStatus, const L0StatusPack& l0Status,
    L1StatusPack& l1Status) const
{
    if (l0Status.kL0 == coreStatus.k) {
        return;
    }
    // 1 -> let k_al1 bigger, 2 -> let k_bl1 bigger, 0 -> no matter
    int32_t kMaxAxis = 0;
    if (!tilingIns_->aType_.isTrans && !tilingIns_->bType_.isTrans) {
        kMaxAxis = 1;
    }
    if (tilingIns_->aType_.isTrans && tilingIns_->bType_.isTrans) {
        kMaxAxis = 2;
    }
    if (!tilingIns_->aType_.isTrans && tilingIns_->bType_.isTrans) {
        kMaxAxis = l0Status.mL0 > l0Status.nL0 ? 1 : 2;
    }
    // Not Support FP32 mode for NZ format and hardware with pre_ub
    if (kMaxAxis != 0) {
        NeitherFullLoadKforND(coreStatus, l0Status, l1Status, kMaxAxis);
    } else {
        NeitherFullLoadKforNZ(coreStatus, l0Status, l1Status);
    }
    if (g_tempCfg.factorSplit) {
        if (l1Status.kAL1 > l1Status.kBL1 && l1Status.kAL1 % l1Status.kBL1 != 0) {
            while (l1Status.kAL1 % l1Status.kBL1 != 0 ||
                (l1Status.kAL1 != l1Status.kBL1 && coreStatus.k % l1Status.kAL1 != 0)) {
                l1Status.kAL1 -= 1;
            }
        }
        if (l1Status.kAL1 < l1Status.kBL1 && l1Status.kBL1 % l1Status.kAL1 != 0) {
            while (l1Status.kBL1 % l1Status.kAL1 != 0 ||
                (l1Status.kAL1 != l1Status.kBL1 && coreStatus.k % l1Status.kBL1 != 0)) {
                l1Status.kBL1 -= 1;
            }
        }
    }
}

void MatmulTilingAlgorithm::L1StatusNeitherFullLoad(const CoreStatusPack& coreStatus, const L0StatusPack& l0Status,
    L1StatusPack& l1Status, int32_t res[][IDX_SEVEN]) const
{
    // if b matrix in L1, then b matrix must full load, skip non-full process
    if (tilingIns_->aType_.pos == TPosition::TSCM || tilingIns_->bType_.pos == TPosition::TSCM) {
        return;
    }
    if (g_tempCfg.l1DB == DB_ON) {
        NeitherFullLoadDb(coreStatus, l0Status, l1Status, DB_ON);
    }
    NeitherFullLoadK(coreStatus, l0Status, l1Status);
    NeitherFullLoadMN(coreStatus, l0Status, l1Status);

    res[IDX_THREE][IDX_ZERO] = l1Status.kAL1;
    res[IDX_THREE][IDX_ONE] = l1Status.mAL1;
    res[IDX_THREE][IDX_TWO] = l1Status.dbAL1;
    res[IDX_THREE][IDX_THREE] = l1Status.kBL1;
    res[IDX_THREE][IDX_FOUR] = l1Status.nBL1;
    res[IDX_THREE][IDX_FIVE] = l1Status.dbBL1;
    res[IDX_THREE][IDX_SIX] = l1Status.loadSize;
}

void MatmulTilingAlgorithm::GetL1Factors(const std::string& opType, const MatmulRunParas& param,
    const CoreStatusPack& coreStatus, const L0StatusPack& l0Status, L1StatusPack& l1Status) const
{
    (void)(opType);
    (void)(param);
    // get mAL1, nBL1, kAL1, kBL1 factors when L0, singlecore factor is know
    // get al1, bl1 double buffer factors
    const int32_t mte1Loop = MIN_MTE1_LOAD /
        ((l0Status.nL0 == 1 ? 1 : l0Status.kL0) + (l0Status.kL0 == 1 ? 1 : l0Status.mL0));
    int32_t res[IDX_FOUR][IDX_SEVEN] = {0};
    l1Status.allTimes = MathUtil::CeilDivision(coreStatus.k, l0Status.kL0);
    l1Status.maxMAL1 = (coreStatus.m + l0Status.mL0 - 1) / l0Status.mL0;
    l1Status.maxNBL1 = (coreStatus.n + l0Status.nL0 - 1) / l0Status.nL0;
    l1Status.maxKAL1 =
        max(mte1Loop, ((MIN_MTE1_LOAD + l0Status.mL0 - 1) / l0Status.mL0 + l0Status.kL0 - 1) / l0Status.kL0);
    l1Status.maxKBL1 =
        max(mte1Loop, ((MIN_MTE1_LOAD + l0Status.nL0 - 1) / l0Status.nL0 + l0Status.kL0 - 1) / l0Status.kL0);
    if (tilingIns_->isSupportL0c2Out && tilingIns_->isBias) {
        l1Status.channelWiseTimes++;
    }
    // both AL1 and Bl1 full load
    int32_t bothFullLoadFactors[L1_FACTORS_LEN] = { coreStatus.k, coreStatus.k, l1Status.maxMAL1,
                                                    l1Status.maxNBL1, DB_OFF, DB_OFF };
    // Need to consider L1 extension in FP32 Mode
    l1Status.SetStatus(bothFullLoadFactors);
    L1StatusBothFullLoad(coreStatus, l0Status, l1Status, res);
    // only AL1 full load
    int32_t al1FullLoadFactors[L1_FACTORS_LEN] = {coreStatus.k, l0Status.kL0, l1Status.maxMAL1, 1, DB_OFF, DB_OFF};
    l1Status.SetStatus(al1FullLoadFactors);
    L1StatusAl1FullLoad(coreStatus, l0Status, l1Status, res);
    // only BL1 full load
    int32_t bl1FullLoadFactors[L1_FACTORS_LEN] = {l0Status.kL0, coreStatus.k, 1, l1Status.maxNBL1, DB_OFF, DB_OFF};
    l1Status.SetStatus(bl1FullLoadFactors);
    L1StatusBl1FullLoad(coreStatus, l0Status, l1Status, res);
    // neither AL1 nor Bl1 full load
    res[IDX_THREE][IDX_SIX] = INT_MAX;
    int32_t neitherFullLoadFactors[L1_FACTORS_LEN] = {l0Status.kL0, l0Status.kL0, 1, 1, DB_ON, DB_ON};
    l1Status.SetStatus(neitherFullLoadFactors);
    L1StatusNeitherFullLoad(coreStatus, l0Status, l1Status, res);
    // choose the final factors
    int32_t* tmpFactors = res[IDX_THREE];
    int32_t tmpLoadSize = tmpFactors[IDX_SIX];
    int32_t reduceSize = static_cast<int32_t>(C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);
    const int32_t kAl1FactorOne = res[IDX_ONE][IDX_ZERO] > 0 ? MathUtil::CeilDivision(
        MathUtil::CeilDivision(GetSingleK(), reduceSize), (coreStatus.kDim * res[IDX_ONE][IDX_ZERO])) :
                                                         1;
    const int32_t kBl1FactorTwo = res[IDX_TWO][IDX_THREE] > 0 ? MathUtil::CeilDivision(
        MathUtil::CeilDivision(GetSingleK(), reduceSize), (coreStatus.kDim * res[IDX_TWO][IDX_THREE])) :
                                                          1;
    const int32_t kAl1FactorZero = res[IDX_ZERO][IDX_ZERO] > 0 ? MathUtil::CeilDivision(
        MathUtil::CeilDivision(GetSingleK(), reduceSize), (coreStatus.kDim * res[IDX_ZERO][IDX_ZERO])) :
                                                           1;
    const int32_t kBl1FactorZero = res[IDX_ZERO][IDX_THREE] > 0 ? MathUtil::CeilDivision(
        MathUtil::CeilDivision(GetSingleK(), reduceSize), (coreStatus.kDim * res[IDX_ZERO][IDX_THREE])) :
                                                            1;

    const bool al1FullLoad = (tilingIns_->aType_.type == CubeFormat::ND && tilingIns_->bType_.type == CubeFormat::ND) ?
        (l1Status.aL1FullLoad && kAl1FactorOne == 1) :
        l1Status.aL1FullLoad;
    const bool bl1FullLoad = (tilingIns_->aType_.type == CubeFormat::ND && tilingIns_->bType_.type == CubeFormat::ND) ?
        (l1Status.bL1FullLoad && kBl1FactorTwo == 1) :
        l1Status.bL1FullLoad;
    const bool bothFullLoad = (tilingIns_->aType_.type == CubeFormat::ND && tilingIns_->bType_.type == CubeFormat::ND) ?
        (l1Status.bothFullLoad && kAl1FactorZero == 1 && kBl1FactorZero == 1) :
        l1Status.bothFullLoad;
    if (al1FullLoad && (res[IDX_ONE][IDX_SIX] < tmpLoadSize || (res[IDX_ONE][IDX_SIX] == tmpLoadSize &&
        res[IDX_ONE][IDX_ONE] + res[IDX_ONE][IDX_FOUR] >= tmpFactors[IDX_ONE] + tmpFactors[IDX_FOUR]))) {
        tmpFactors = res[IDX_ONE];
        tmpLoadSize = tmpFactors[IDX_SIX];
        TILING_LOG_DEBUG("Select Mode One.");
    }
    if (bl1FullLoad && (res[IDX_TWO][IDX_SIX] < tmpLoadSize || (res[IDX_TWO][IDX_SIX] == tmpLoadSize &&
        res[IDX_TWO][IDX_ONE] + res[IDX_TWO][IDX_FOUR] >= tmpFactors[IDX_ONE] + tmpFactors[IDX_FOUR]))) {
        tmpFactors = res[IDX_TWO];
        tmpLoadSize = tmpFactors[IDX_SIX];
        TILING_LOG_DEBUG("Select Mode Two.");
    }
    if (bothFullLoad && (res[IDX_ZERO][IDX_SIX] < tmpLoadSize || (res[IDX_ZERO][IDX_SIX] == tmpLoadSize &&
        res[IDX_ZERO][IDX_ONE] + res[IDX_ZERO][IDX_FOUR] >= tmpFactors[IDX_ONE] + tmpFactors[IDX_FOUR]))) {
        tmpFactors = res[IDX_ZERO];
        TILING_LOG_DEBUG("Select Mode Zero.");
    }
    int32_t resL1Factors[L1_FACTORS_LEN] = {tmpFactors[IDX_ZERO], tmpFactors[IDX_THREE], tmpFactors[IDX_ONE],
                                            tmpFactors[IDX_FOUR], tmpFactors[IDX_TWO], tmpFactors[IDX_FIVE]};
    l1Status.SetStatus(resL1Factors);
}

void MatmulTilingAlgorithm::GetUsedSize(int32_t& l1Size, int32_t& l0cSize, int32_t& ubSize,
                                        int32_t a1LengthCache, int32_t b1LengthCache) const
{
    const uint32_t aTypeSize = DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType);
    const uint32_t bTypeSize = DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType);
    const uint32_t cTypeSize = DTYPE_BYTE_TAB.at(tilingIns_->cType_.dataType);
    const uint32_t biasTypeSize = DTYPE_BYTE_TAB.at(tilingIns_->biasType_.dataType);

    const int32_t a1Length = tilingIns_->tiling_.get_baseM() * tilingIns_->tiling_.get_baseK() * aTypeSize / BITS_PER_BYTE;
    const int32_t b1Length = tilingIns_->tiling_.get_baseN() * tilingIns_->tiling_.get_baseK() * bTypeSize / BITS_PER_BYTE;
    const int32_t c1Length = tilingIns_->tiling_.get_baseN() * tilingIns_->tiling_.get_baseM() * FP32_BYTES;

    if (tilingIns_->aType_.pos != TPosition::TSCM) {
        l1Size += tilingIns_->tiling_.get_depthA1() * a1Length;
        if (tilingIns_->enableL1CacheUB && tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
            l1Size += tilingIns_->tiling_.get_depthAL1CacheUB() * a1LengthCache;
        }
    }
    if (tilingIns_->bType_.pos != TPosition::TSCM) {
        l1Size += tilingIns_->tiling_.get_depthB1() * b1Length;
        if (tilingIns_->enableL1CacheUB && tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
            l1Size += tilingIns_->tiling_.get_depthBL1CacheUB() * b1LengthCache;
        }
    }

    l0cSize += c1Length;

    if (static_cast<bool>(tilingIns_->tiling_.get_isBias())) {
        if ((tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND910B ||
            tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND310B) &&
            tilingIns_->biasType_.pos != TPosition::TSCM) {
            // for ascend910b1 bias: gm -> l1 -> bt, need extra l1 space, support bias transform
            l1Size += tilingIns_->tiling_.get_baseN() * biasTypeSize;
        }
    }

    // in v100/v200, nd2nz and nz2nd was simulated with intrins, need extra ub space
    // in V300, nd2nz was simulated with intrins, need extra ub space
    if (tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND910 ||
        tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND310P ||
        tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND310B) {
        // case2: input ND(GM/VECCALC), ND -> NZ transform, for now A/B reuse, only process with tail block, need UB space
        // (1) input GM, format is ND, need do zero-fill to non-aligned tail block in ub
        // (2) input VECCALC, format is ND, need do zero-fill to non-aligned tail block in ub
        int32_t aUbLength = 0;
        int32_t bUbLength = 0;
        if (!tilingIns_->aType_.isTrans && ((tilingIns_->tiling_.get_singleCoreK() * aTypeSize / BITS_PER_BYTE) % C0_BYTE_SIZE != 0)) {
            aUbLength = tilingIns_->tiling_.get_baseM() * C0_BYTE_SIZE;
        }
        if (tilingIns_->aType_.isTrans &&
            ((tilingIns_->tiling_.get_singleCoreM() * aTypeSize / BITS_PER_BYTE) % C0_BYTE_SIZE != 0)) {
            aUbLength = tilingIns_->tiling_.get_baseK() * C0_BYTE_SIZE;
        }

        if (!tilingIns_->bType_.isTrans && ((tilingIns_->tiling_.get_singleCoreN() * bTypeSize / BITS_PER_BYTE) % C0_BYTE_SIZE != 0)) {
            bUbLength = tilingIns_->tiling_.get_baseK() * C0_BYTE_SIZE;
        }
        if (tilingIns_->bType_.isTrans &&
            ((tilingIns_->tiling_.get_singleCoreK() * bTypeSize / BITS_PER_BYTE) % C0_BYTE_SIZE != 0)) {
            bUbLength = tilingIns_->tiling_.get_baseN() * C0_BYTE_SIZE;
        }
        if (tilingIns_->aType_.pos == TPosition::TSCM) {
            aUbLength = 0;
        }
        if (tilingIns_->bType_.pos == TPosition::TSCM) {
            bUbLength = 0;
        }

        if ((tilingIns_->aType_.type == CubeFormat::ND || tilingIns_->bType_.type == CubeFormat::ND)) {
            ubSize += max(aUbLength, bUbLength);
        }

        // V300 only needs extra buffer when INPUT are at GM/UB.
        if (tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND310B) {
            return;
        }

        // case3: output GM/VECCAL, format is ND, for now not re-use input and output non-aligned, is related with db open
        // (1) output GM, format is NZ/ND_ALIGN/ND, need restore in ub, ND and D is non-aligned , then add more 32B, ub->gm NZ->ND format and data type transform
        // (2) output VECCALC,format is ND_ALIGN/ND(D aligned), need doNZ->ND transform in ub
        if (tilingIns_->cType_.pos == TPosition::GM) {
            ubSize += tilingIns_->tiling_.get_baseM() * tilingIns_->tiling_.get_baseN() * cTypeSize;
            if (tilingIns_->cType_.type == CubeFormat::ND &&
                (tilingIns_->tiling_.get_singleCoreN() * cTypeSize) % C0_BYTE_SIZE != 0) {
                ubSize += C0_BYTE_SIZE;
            }
        }
        if (tilingIns_->cType_.pos == TPosition::VECCALC && tilingIns_->cType_.type != CubeFormat::NZ) {
            ubSize += tilingIns_->tiling_.get_baseM() * tilingIns_->tiling_.get_baseN() * cTypeSize;
        }
        // for V200/V100, if setquanttensor and output is not nd, need extra ubsize for copy tensor from gm to ub
        if (tilingIns_->deqType == DequantType::TENSOR && tilingIns_->cType_.type == CubeFormat::NZ) {
            ubSize += static_cast<int32_t>(tilingIns_->tiling_.get_baseN() * DTYPE_BYTE_TAB.at(DataType::DT_UINT64));
        }
    }
    return;
}

void MatmulTilingAlgorithm::GetBankConflictSize(const L1StatusPack& l1Status, const L0StatusPack& l0Status,
    int32_t& length, bool isAMatrix) const
{
    constexpr int blockSize = 32;
    constexpr int bankLen = 512;
    bool isBankConflict = false;
    int bankConflictSize = 0;
    const int32_t reduceSize = static_cast<int32_t>(C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);
    if (isAMatrix) {
        if (tilingIns_->aType_.isTrans) {
            isBankConflict =
                MathUtil::CeilDivision(l1Status.mAL1 * l0Status.mL0 * C0_SIZE, C0_SIZE) *
                blockSize % bankLen ==
                0 ?
                true :
                false;
            bankConflictSize = l0Status.kL0 * reduceSize * C0_SIZE *
                MathUtil::CeilDivision(l1Status.kAL1, l0Status.kL0) *
                DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
        } else {
            isBankConflict =
                MathUtil::CeilDivision(MathUtil::CeilDivision(l1Status.kAL1, l0Status.kL0) * l0Status.kL0 * reduceSize,
                C0_SIZE) * blockSize % bankLen ==
                0 ?
                true :
                false;
            bankConflictSize = l0Status.mL0 * C0_SIZE * C0_SIZE * l1Status.mAL1 *
                DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
        }
    } else {
        if (tilingIns_->bType_.isTrans) {
            isBankConflict =
                MathUtil::CeilDivision(MathUtil::CeilDivision(l1Status.kBL1, l0Status.kL0) * l0Status.kL0 * reduceSize,
                C0_SIZE) * blockSize % bankLen ==
                0 ?
                true :
                false;
            bankConflictSize = l0Status.nL0 * C0_SIZE * C0_SIZE * l1Status.nBL1 *
                DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
        } else {
            isBankConflict =
                MathUtil::CeilDivision(l1Status.nBL1 * l0Status.nL0 * C0_SIZE, C0_SIZE) *
                blockSize % bankLen ==
                0 ?
                true :
                false;
            bankConflictSize = l0Status.kL0 * reduceSize * C0_SIZE *
                MathUtil::CeilDivision(l1Status.kBL1, l0Status.kL0) *
                DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
        }
    }
    if (isBankConflict) {
        length = length + bankConflictSize;
    }
}

void MatmulTilingAlgorithm::GetBankConflictSize(int32_t& length, bool isAMatrix) const
{
    constexpr int blockSize = 32;
    constexpr int bankLen = 512;
    bool isBankConflict = false;
    int bankConflictSize = 0;
    if (isAMatrix) {
        if (tilingIns_->aType_.isTrans) {
            isBankConflict =
                MathUtil::CeilDivision(tilingIns_->tiling_.get_stepM() * tilingIns_->tiling_.get_baseM(), C0_SIZE) *
                blockSize % bankLen ==
                0 ?
                true :
                false;
            bankConflictSize = tilingIns_->tiling_.get_baseK() * C0_SIZE * tilingIns_->tiling_.get_stepKa() *
                DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
        } else {
            isBankConflict =
                MathUtil::CeilDivision(tilingIns_->tiling_.get_stepKa() * tilingIns_->tiling_.get_baseK(), C0_SIZE) *
                blockSize % bankLen ==
                0 ?
                true :
                false;
            bankConflictSize = tilingIns_->tiling_.get_baseM() * C0_SIZE * tilingIns_->tiling_.get_stepM() *
                DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
        }
    } else {
        if (tilingIns_->bType_.isTrans) {
            isBankConflict =
                MathUtil::CeilDivision(tilingIns_->tiling_.get_stepKb() * tilingIns_->tiling_.get_baseK(), C0_SIZE) *
                blockSize % bankLen ==
                0 ?
                true :
                false;
            bankConflictSize = tilingIns_->tiling_.get_baseN() * C0_SIZE * tilingIns_->tiling_.get_stepN() *
                DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
        } else {
            isBankConflict =
                MathUtil::CeilDivision(tilingIns_->tiling_.get_stepN() * tilingIns_->tiling_.get_baseN(), C0_SIZE) *
                blockSize % bankLen ==
                0 ?
                true :
                false;
            bankConflictSize = tilingIns_->tiling_.get_baseK() * C0_SIZE * tilingIns_->tiling_.get_stepKb() *
                DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
        }
    }
    if (isBankConflict) {
        length = length + bankConflictSize;
    }
}

int32_t MatmulTilingAlgorithm::GetAL1UbSize(const L1StatusPack& l1Status, const L0StatusPack& l0Status) const
{
    int32_t a1Length = 0;
    const int32_t reduceSize = static_cast<int32_t>(C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);
    if (IsUbNd2Nz()) {
        // A matrix ND2NZ
        if (tilingIns_->aType_.type == CubeFormat::ND) {
            a1Length = l0Status.mL0 * C0_SIZE * l0Status.kL0 * reduceSize *
                DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
            if (tilingIns_->mmConfigType == 1) {
                a1Length = a1Length * MathUtil::CeilDivision(l1Status.kAL1, l0Status.kL0) * l1Status.mAL1;
            }
            // bank conflict
            GetBankConflictSize(l1Status, l0Status, a1Length, true);
        }
    }
    return a1Length;
}

int32_t MatmulTilingAlgorithm::GetBL1UbSize(const L1StatusPack& l1Status, const L0StatusPack& l0Status) const
{
    int32_t b1Length = 0;
    const int32_t reduceSize = static_cast<int32_t>(C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);
    if (IsUbNd2Nz()) {
        // B matrix ND2NZ
        if (tilingIns_->bType_.type == CubeFormat::ND) {
            b1Length = l0Status.nL0 * C0_SIZE * l0Status.kL0 * reduceSize *
                DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
            if (tilingIns_->mmConfigType == 1) {
                b1Length = b1Length * MathUtil::CeilDivision(l1Status.kBL1, l0Status.kL0) * l1Status.nBL1;
            }
            // bank conflict
            GetBankConflictSize(l1Status, l0Status, b1Length, false);
        }
    }
    return b1Length;
}

bool MatmulTilingAlgorithm::IsUbNd2Nz() const
{
    if (tilingIns_->enVecND2NZ && tilingIns_->mmConfigType == 1 &&
        tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        return true;
    }
    return false;
}

void MatmulTilingAlgorithm::GetTransLength(int32_t& transLength) const
{
    int32_t a1Length = 0;
    int32_t b1Length = 0;
    int32_t c1Length = 0;
    int32_t biasLength = 0;
    if (tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND910 ||
        tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND310P ||
        tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND310B) {
        // A matrix ND2NZ
        if (tilingIns_->aType_.type == CubeFormat::ND) {
            a1Length = tilingIns_->tiling_.get_baseM() * tilingIns_->tiling_.get_baseK() *
                DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
            if (tilingIns_->mmConfigType == 1) {
                a1Length = a1Length * tilingIns_->tiling_.get_stepKa() * tilingIns_->tiling_.get_stepM();
            }
            // bank conflict
            GetBankConflictSize(a1Length, true);
        }
        // B matrix ND2NZ
        if (tilingIns_->bType_.type == CubeFormat::ND
            || (DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8) &&
            tilingIns_->bType_.type == CubeFormat::NZ && tilingIns_->bType_.isTrans == false)) {
            b1Length = tilingIns_->tiling_.get_baseN() * tilingIns_->tiling_.get_baseK() *
                DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
            if (tilingIns_->mmConfigType == 1) {
                b1Length = b1Length * tilingIns_->tiling_.get_stepKb() * tilingIns_->tiling_.get_stepN();
            }
            // bank conflict
            GetBankConflictSize(b1Length, false);
        }
        // C matrix NZ2ND
        if (tilingIns_->cType_.type == CubeFormat::ND || tilingIns_->cType_.pos == TPosition::GM) {
            c1Length = tilingIns_->tiling_.get_baseN() * tilingIns_->tiling_.get_baseM() *
                DTYPE_BYTE_TAB.at(tilingIns_->cType_.dataType);
        }
        // Bias
        if (tilingIns_->isBias && tilingIns_->biasType_.pos != TPosition::VECCALC) {
            biasLength = tilingIns_->tiling_.get_baseN() * DTYPE_BYTE_TAB.at(tilingIns_->biasType_.dataType);
        }
        // quant tensor
        if (DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8)) {
            int32_t quantLength = tilingIns_->tiling_.get_baseN() * sizeof(uint64_t);
            biasLength = max(quantLength, biasLength);
        }
    }

    transLength = max(max(a1Length, b1Length), max(c1Length, biasLength));
}

bool MatmulTilingAlgorithm::CheckBaseMN() const
{
    // check bias table
    if ((tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND910B ||
        tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND310B) &&
        tilingIns_->isBias && (tilingIns_->baseN > MAX_BIAS_N * C0_SIZE) && tilingIns_->isSupportL0c2Out) {
        return false;
    }
    if (tilingIns_->baseM != -1 && tilingIns_->baseN != -1) {
        return (tilingIns_->baseM * tilingIns_->baseN * FP32_BYTES <= tilingIns_->bufferPool_.l0CSize &&
            tilingIns_->baseM * C0_BYTE_SIZE <= tilingIns_->bufferPool_.l0ASize &&
            tilingIns_->baseN * C0_BYTE_SIZE <= tilingIns_->bufferPool_.l0BSize);
    }
    if (tilingIns_->baseM != -1) {
        return (tilingIns_->baseM * C0_SIZE * FP32_BYTES <= tilingIns_->bufferPool_.l0CSize &&
            tilingIns_->baseM * C0_BYTE_SIZE <= tilingIns_->bufferPool_.l0ASize);
    }
    if (tilingIns_->baseN != -1) {
        return (tilingIns_->baseN * C0_SIZE * FP32_BYTES <= tilingIns_->bufferPool_.l0CSize &&
            tilingIns_->baseN * C0_BYTE_SIZE <= tilingIns_->bufferPool_.l0BSize);
    }
    return true;
}

int32_t MatmulTilingAlgorithm::GetIteratorOrder(const SingleCoreStatus& singleCoreStatus, const int32_t singleCoreM,
    const int32_t singleCoreN, const int32_t singleCoreK) const
{
    if (tilingIns_->traverse_ != MatrixTraverse::NOSET) {
        return static_cast<int32_t>(tilingIns_->traverse_) - 1;
    }
    const int32_t reduceSize = static_cast<int32_t>(C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);
    const bool fullkAL1Load =
        (static_cast<float>(singleCoreK) / (singleCoreStatus.l1Status.kAL1 * reduceSize)) > 1.0 ? false : true;
    bool fullkBL1Load =
        (static_cast<float>(singleCoreK) / (singleCoreStatus.l1Status.kBL1 * reduceSize)) > 1.0 ? false : true;

    // if KAL1 and KBL1 both can not be full loaded, then select m or n which is no matter
    if (!fullkAL1Load && !fullkBL1Load) {
        return static_cast<int32_t>(MatrixTraverse::FIRSTM) - 1;
    } else if (fullkAL1Load && !fullkBL1Load) { // if KAL1 is full loaded, then select the order N first
        return static_cast<int32_t>(MatrixTraverse::FIRSTN) - 1;
    } else if (!fullkAL1Load && fullkBL1Load) { // if KBL1 is full loaded, then select the order M first
        return static_cast<int32_t>(MatrixTraverse::FIRSTM) - 1;
    } else {
        // if AL1LoadSize less then BL1LoadSize, then select order N first, vice versa.
        const int32_t mLoop = MathUtil::CeilDivision(singleCoreM,
            singleCoreStatus.l1Status.mAL1 * singleCoreStatus.l0Status.mL0 * C0_SIZE);
        const int32_t nLoop = MathUtil::CeilDivision(singleCoreN,
            singleCoreStatus.l1Status.nBL1 * singleCoreStatus.l0Status.nL0 * C0_SIZE);
        const int32_t aL1LoadSize = singleCoreM + singleCoreN * mLoop;
        const int32_t bL1LoadSize = singleCoreN + singleCoreM * nLoop;
        return aL1LoadSize < bL1LoadSize ? 1 : 0;
    }
}

void MatmulTilingAlgorithm::UpdateDimCalculator(DimCalculator& dimCalRes) const
{
    if (dimCalRes.totalLoadSize > dimCalRes.tmpLoadSize) {
        dimCalRes.bmatSize = dimCalRes.tmpBmatSize;
        dimCalRes.amatSize = dimCalRes.tmpAmatSize;
        dimCalRes.totalLoadSize = dimCalRes.tmpLoadSize;
        dimCalRes.tmpValue = 0;
    }
}

void MatmulTilingAlgorithm::CalcLoadSize(const DimFactor& dimFactor, const CoreStatusPack& coreStatus,
    DimCalculator& dimCalRes, const MatmulRunParas& params) const
{
    dimCalRes.totalLoadSize = INT_MAX;
    // A/B fullLoad or A fullLoad + B Kdim fullLoad or B fullLoad + A Kdim fullLoad(1/2/4)
    const int32_t totalSize = dimCalRes.amatSize + dimCalRes.bmatSize; // batch==1
    constexpr int32_t minMNSize = 16;
    constexpr int32_t minKSize = 64;
    constexpr int32_t minTotalSize = 128;
    const int32_t n0 = min(minMNSize, coreStatus.n); // need check m,n > 16 or m,n<16
    const int32_t m0 = min(minMNSize, ((n0 == 0) ? 0 : min(coreStatus.m, minTotalSize / n0)));
    const int32_t k0 = (m0 != 0 && n0 != 0) ? 
        min(min(minKSize / m0, minKSize / n0), coreStatus.k) : coreStatus.k;
    const int32_t dbBuffer = 2;

    // A/B fullLoad or A fullLoad + B Kdim fullLoad or B fullLoad + A Kdim fullLoad(1/2/4)
    // loadsize = K*(N*mdim+M*ndim)
    const bool bothFullLoad = static_cast<int64_t>(totalSize) * static_cast<int64_t>(dimCalRes.kBytes) <=
        static_cast<int64_t>(tilingIns_->bufferPool_.l1Size);
    const bool afullLoadPlsBKFullLoad =
        static_cast<int64_t>(dimCalRes.amatSize + n0 * dbBuffer) * static_cast<int64_t>(dimCalRes.kBytes) <=
        static_cast<int64_t>(tilingIns_->bufferPool_.l1Size);
    const bool bfullLoadPlsaKFullLoad =
        static_cast<int64_t>(dimCalRes.bmatSize + m0 * dbBuffer) * static_cast<int64_t>(dimCalRes.kBytes) <=
        static_cast<int64_t>(tilingIns_->bufferPool_.l1Size);
    if (afullLoadPlsBKFullLoad || bfullLoadPlsaKFullLoad || bothFullLoad) {
        dimCalRes.tmpAmatSize = dimCalRes.oriAmatSize * dimFactor.n;
        dimCalRes.tmpBmatSize = dimCalRes.oriBmatSize * dimFactor.m;
        dimCalRes.tmpLoadSize = dimCalRes.tmpAmatSize + dimCalRes.tmpBmatSize;
        UpdateDimCalculator(dimCalRes);
        return;
    }

    // A kdim not fullLoad + B kdim not fullLoad(9)
    // loadsize = M*K*N*(1/m0+1/n0)
    const bool aKNotfullLoadPlsbKNotFullLoad =
        (n0 * dimCalRes.kBytes + m0 * k0 * C0_SIZE * C0_BYTE_SIZE) * dbBuffer >
        tilingIns_->bufferPool_.l1Size &&
        (m0 * dimCalRes.kBytes + n0 * k0 * C0_SIZE * C0_BYTE_SIZE) * dbBuffer >
        tilingIns_->bufferPool_.l1Size;
    if (aKNotfullLoadPlsbKNotFullLoad) {
        dimCalRes.tmpAmatSize = dimCalRes.oriAmatSize * MathUtil::CeilDivision(params.n32, n0);
        dimCalRes.tmpBmatSize = dimCalRes.oriBmatSize * MathUtil::CeilDivision(params.m32, m0);
        dimCalRes.tmpLoadSize = dimCalRes.tmpAmatSize + dimCalRes.tmpBmatSize;
        UpdateDimCalculator(dimCalRes);
        return;
    }

    // A kdim fullLoad + B kdim fullLoad(5)
    // M*K*(ndim+N/m1) or N*K*(mdim+M/n1)
    const bool aKfullLoadPlsbKFullLoad = (m0 + n0) * dimCalRes.kBytes * dbBuffer <= tilingIns_->bufferPool_.l1Size;
    if (aKfullLoadPlsbKFullLoad) {
        const int32_t m1 = MathUtil::CeilDivision((tilingIns_->bufferPool_.l1Size - n0 *
            dimCalRes.kBytes * dbBuffer), (dimCalRes.kBytes * dbBuffer * m0)) * m0;
        const int32_t n1 = MathUtil::CeilDivision((tilingIns_->bufferPool_.l1Size - m0 *
            dimCalRes.kBytes * dbBuffer), (dimCalRes.kBytes * dbBuffer * n0)) * n0;
        const int32_t mfirstLoad =
            dimCalRes.oriAmatSize * dimFactor.n + dimCalRes.oriBmatSize * MathUtil::CeilDivision(params.m32, m1);
        int32_t nfirstLoad =
            dimCalRes.oriBmatSize * dimFactor.m + dimCalRes.oriAmatSize * MathUtil::CeilDivision(params.n32, n1);
        if (mfirstLoad < nfirstLoad) {
            dimCalRes.tmpAmatSize = dimCalRes.oriAmatSize * dimFactor.n;
            dimCalRes.tmpBmatSize = dimCalRes.oriBmatSize * MathUtil::CeilDivision(params.m32, m1);
        } else {
            dimCalRes.tmpAmatSize = dimCalRes.oriAmatSize * MathUtil::CeilDivision(params.n32, n1);
            dimCalRes.tmpBmatSize = dimCalRes.oriBmatSize * dimFactor.m;
        }
        dimCalRes.tmpLoadSize = dimCalRes.tmpAmatSize + dimCalRes.tmpBmatSize;
        UpdateDimCalculator(dimCalRes);
        return;
    }

    //  A fullLoad + B Kdim not fullLoad or A K fullLoad + B Kdim not fullLoad(3/6)
    // mdim = coreNum; ndim = 1；
    // loadsize = M*K*(ndim+N/m0)
    const bool afullLoadPlsbKNotFullLoad = (dimCalRes.amatSize * dimCalRes.kBytes +
        n0 * k0 * C0_SIZE * C0_BYTE_SIZE * dbBuffer) <= tilingIns_->bufferPool_.l1Size;
    const bool aKfullLoadPlsbKNotFullLoad = (m0 * dimCalRes.kBytes * dbBuffer +
        n0 * k0 * C0_SIZE * C0_BYTE_SIZE * dbBuffer) <= tilingIns_->bufferPool_.l1Size;
    if (afullLoadPlsbKNotFullLoad || aKfullLoadPlsbKNotFullLoad) {
        dimCalRes.tmpAmatSize = dimCalRes.oriAmatSize * dimFactor.n;
        dimCalRes.tmpBmatSize = dimCalRes.oriBmatSize * MathUtil::CeilDivision(params.m32, m0);
        dimCalRes.tmpLoadSize = dimCalRes.tmpAmatSize + dimCalRes.tmpBmatSize;
        UpdateDimCalculator(dimCalRes);
    }

    // A kdim not fullLoad + B fullLoad or A kdim not fullLoad + B kdim fullLoad(7/8)
    // loadsize = N*K*(mdim+M/n0)
    const bool aKNotfullLoadPlsbFullLoad = (dimCalRes.bmatSize * dimCalRes.kBytes +
        m0 * k0 * C0_SIZE * C0_BYTE_SIZE * dbBuffer) <= tilingIns_->bufferPool_.l1Size;
    const bool aKNotfullLoadPlsbKFullLoad = (n0 * dimCalRes.kBytes * dbBuffer +
        m0 * k0 * C0_SIZE * C0_BYTE_SIZE * dbBuffer) <= tilingIns_->bufferPool_.l1Size;
    if (aKNotfullLoadPlsbFullLoad || aKNotfullLoadPlsbKFullLoad) {
        dimCalRes.tmpAmatSize = dimCalRes.oriBmatSize * dimFactor.m;
        dimCalRes.tmpBmatSize = dimCalRes.oriAmatSize * MathUtil::CeilDivision(params.n32, n0);
        dimCalRes.tmpLoadSize = dimCalRes.tmpAmatSize + dimCalRes.tmpBmatSize;
        UpdateDimCalculator(dimCalRes);
    }
}

int32_t MatmulTilingAlgorithm::LoopNumFromSingleCoreToL0(const CoreStatusPack& coreStatus,
    const DimFactor& dimFactor) const
{
    if (!dimFactor.IsValid()) {
        return 0;
    }
    constexpr int32_t minTotalSize = 128;
    constexpr int32_t minSize = 64;
    constexpr int32_t minN0Size = 16;
    int32_t n0 = min(min(minN0Size, coreStatus.n), minSize);
    int32_t m0 = (n0 == 0) ? 0 : min(min(coreStatus.m, minTotalSize / n0), minSize);
    n0 = (m0 == 0) ? 0 : min(min(coreStatus.n, minTotalSize / m0), minSize);
    m0 = (n0 == 0) ? 0 : min(min(coreStatus.m, minTotalSize / n0), minSize);
    const int32_t k0 = (m0 != 0 && n0 != 0) ?
        min(min(minSize / m0, minSize / n0), coreStatus.k) : coreStatus.k;
    const int32_t loopNum = MathUtil::CeilDivision(coreStatus.m, m0) * MathUtil::CeilDivision(coreStatus.n, n0) *
        MathUtil::CeilDivision(coreStatus.k, k0);
    return loopNum;
}

int32_t MatmulTilingAlgorithm::GetBigPackageCondition(const CoreStatusPack &coreStatus,
    const DimCalculator &dimCalRes, const MatmulRunParas &params) const
{
    if (tilingIns_->bType_.isTrans == true && tilingIns_->aType_.isTrans == false) {
        return ATTACH_FLAG_ZERO;
    }
    const int minSize = 16;
    bool flag = true;
    if (tilingIns_->bType_.isTrans == false) {
        if (params.n32 >= minSize && coreStatus.n < minSize) {
            flag = false;
        }
    }
    if (tilingIns_->aType_.isTrans) {
        if (params.m32 >= minSize && coreStatus.m < minSize) {
            flag = false;
        }
    }

    if (!dimCalRes.bigPackage && !flag) {
        return ATTACH_FLAG_ZERO;
    } else if (!dimCalRes.bigPackage && flag) {
        return ATTACH_FLAG_TWO;
    } else if (dimCalRes.bigPackage && !flag) {
        return ATTACH_FLAG_ONE;
    } else {
        return ATTACH_FLAG_ZERO;
    }
}

void MatmulTilingAlgorithm::GetDimsHelper(const DimFactor& dimFactor, CoreStatusPack& coreStatus,
    DimCalculator& dimCalRes, const MatmulRunParas& params)
{
    dimCalRes.kNum = (dimFactor.k == 0) ? 0 : (params.k32 / dimFactor.k * C0_SIZE * REDUCE_BLOCK_SIZE); // contain k * 16
    dimCalRes.kBytes = dimCalRes.kNum * INPUTDTYPE_BYTES;                 // contain k * 16 * 2
    coreStatus.batch = MathUtil::CeilDivision(params.batch32, dimFactor.batch);
    coreStatus.m = MathUtil::CeilDivision(params.m32, dimFactor.m);
    coreStatus.n = MathUtil::CeilDivision(params.n32, dimFactor.n);
    coreStatus.k = (dimFactor.k == 0) ? 0 : (params.k32 / dimFactor.k);
    if (tilingIns_->enableSplitK_) {
        if (params.kMapped != params.k32) { // need check--splitK
            dimCalRes.kNum = params.kMapped / dimFactor.k * NUM_TWO * C0_SIZE * REDUCE_BLOCK_SIZE;
            coreStatus.k = params.kMapped / dimFactor.k * NUM_TWO;
        }
    }

    // load size of A matrix is batch * m
    // load size of B matrix is n
    dimCalRes.oriAmatSize = params.batch32 * params.m32;
    dimCalRes.oriBmatSize = params.oriShapeBbatch > 1 ? params.batch32 * params.n32 : params.n32;
    dimCalRes.amatSize = coreStatus.batch * coreStatus.m;
    dimCalRes.bmatSize = params.oriShapeBbatch > 1 ? coreStatus.batch * coreStatus.n : coreStatus.n;
    dimCalRes.tmpValue = 0;
    CalcLoadSize(dimFactor, coreStatus, dimCalRes, params);
    if (tilingIns_->enableSplitK_) {
        dimCalRes.totalLoadSize *= coreStatus.k;
    }

    // updateSolution: bool whether update to a new block factor solution
    // has smaller LoadSize or the same LoadSize but batch
    const int bigpackageFlag = GetBigPackageCondition(coreStatus, dimCalRes, params);
    const bool updateConditionBp = bigpackageFlag == 0 ? false : true;
    bool updateConditionBp2 = bigpackageFlag == 2 ? true : false;
    bool updateConditionBp3 = bigpackageFlag == 1 ? false : true;

    const int32_t loopNum = LoopNumFromSingleCoreToL0(coreStatus, dimFactor);
    const bool updateConditionCoreUsed = (!updateConditionBp) && ((loopNum < dimCalRes.loopNumToL0) ||
        (dimFactor.ReduceMul() > dimCalRes.coreUse && loopNum == dimCalRes.loopNumToL0));
    const bool updateConditionLoadsize = (!updateConditionCoreUsed && dimFactor.ReduceMul() == dimCalRes.coreUse) &&
        dimCalRes.totalLoadSize < dimCalRes.minLoadSize;
    const int32_t orgBatchM = params.oriShapeAbatch > 1 ? dimCalRes.batchDimFactor : dimCalRes.mDimFactor;
    const int32_t curBatchM = params.oriShapeAbatch > 1 ? dimFactor.batch : dimFactor.m;
    const bool updateConditionBatchNDim = (!updateConditionCoreUsed && dimFactor.ReduceMul() == dimCalRes.coreUse &&
        dimCalRes.totalLoadSize == dimCalRes.minLoadSize) &&
        ((dimCalRes.nDimFactor * orgBatchM < curBatchM * dimFactor.n) ||
        (dimCalRes.nDimFactor * orgBatchM == curBatchM * dimFactor.n &&
        dimCalRes.batchDimFactor < dimFactor.batch));

    const bool policyCondition =
        UserPolicy(tilingIns_->bType_.pos == TPosition::TSCM ? TilingPolicy::FIXED_B_TSCM : TilingPolicy::NO_POLICY,
        coreStatus, dimCalRes);
    if ((updateConditionBp2 || updateConditionCoreUsed || updateConditionLoadsize || updateConditionBatchNDim) &&
        policyCondition && updateConditionBp3) {
        dimCalRes.minLoadSize = dimCalRes.totalLoadSize;
        dimCalRes.nDimFactor = dimFactor.n;
        dimCalRes.batchDimFactor = dimFactor.batch;
        dimCalRes.mDimFactor = dimFactor.m;
        dimCalRes.kDimFactor = dimFactor.k;
        dimCalRes.coreUse = dimFactor.ReduceMul();
        dimCalRes.loopNumToL0 = loopNum;
        dimCalRes.finalValue = dimCalRes.tmpValue;
        const int32_t minSize = 16;
        dimCalRes.bigPackage = (!tilingIns_->bType_.isTrans ? coreStatus.n >= minSize : true) &&
            (tilingIns_->aType_.isTrans ? coreStatus.m >= minSize : true) && (dimFactor.n * dimFactor.m * dimFactor.k > 1);
        splitCoreFlag_ = true;
    }
}

bool MatmulTilingAlgorithm::UserPolicy(const TilingPolicy policy, const CoreStatusPack& coreStatus,
    const DimCalculator& dimCalRes) const
{
    constexpr int32_t minMNSize = 16;
    constexpr int32_t minKSize = 64;
    constexpr int32_t minTotalSize = 128;
    const int32_t n0 = min(minMNSize, coreStatus.n); // need check m,n > 16 or m,n<16
    const int32_t m0 = min(minMNSize, ((n0 == 0) ? 0 : min(coreStatus.m, minTotalSize / n0)));
    const int32_t k0 = (m0 != 0 && n0 != 0) ? min(min(minKSize / m0, minKSize / n0), coreStatus.k) : coreStatus.k;

    if (policy == TilingPolicy::FIXED_B_TSCM) {
        const int32_t alignFactor = MathUtil::CeilDivision(tilingIns_->alignSingleN, C0_SIZE);
        if (coreStatus.n < alignFactor) {
            return false;
        }
        const int32_t alignNLength = MathUtil::Align(coreStatus.n, alignFactor);
        const int32_t bMatrixSize = alignNLength * dimCalRes.kBytes * 2;
        int32_t aMatrixSize = m0 * k0 * C0_SIZE * C0_BYTE_SIZE;
        int32_t biasSize = 0;
        if (tilingIns_->isSupportL0c2Out && tilingIns_->isBias) {
            biasSize = alignNLength * C0_SIZE * DTYPE_BYTE_TAB.at(tilingIns_->biasType_.dataType);
        }
        if (bMatrixSize + aMatrixSize + biasSize <= tilingIns_->bufferPool_.l1Size) {
            return true;
        } else {
            return false;
        }
    } else if (policy == TilingPolicy::FIXED_A_TSCM) {
        return false;
    } else if (policy == TilingPolicy::FIXED_A_B_TSCM) {
        return false;
    } else {
        return true;
    }
}

bool MatmulTilingAlgorithm::PreProcessMiniShape(const std::string& opType, CoreStatusPack& coreStatus,
    MatmulRunParas& params, const int32_t& coreNum, bool splitKFlag) const
{
    (void)(opType);
    // experience value for mini shape
    const int32_t miniL0cThreshold = tilingIns_->bufferPool_.l0CSize / MIN_FRACTAL_SIZE / FP32_BYTES;
    const int32_t miniL0abThreshold = tilingIns_->bufferPool_.l0ASize / (C0_SIZE * C0_BYTE_SIZE);
    // tend to use less cores for shapes with batch less than coreNum and m/k/n can full load in
    // aicore buffers split_k is conflict with m/n shift_inwards
    bool specialScenario = false;
    if (params.n32 > MIN_MTE1_LOAD) {
        specialScenario = specialScenario ||
            (splitKFlag && ((static_cast<uint32_t>(params.nMapped) & static_cast<uint32_t>(MIN_MTE1_LOAD - 1)) != 0));
    }
    if (params.m32 > MIN_MTE1_LOAD) {
        specialScenario = specialScenario ||
            (splitKFlag && ((static_cast<uint32_t>(params.mMapped) & static_cast<uint32_t>(MIN_MTE1_LOAD - 1)) != 0));
    }

    if (params.batch32 * params.n32 * params.m32 <= coreNum && params.m32 * params.k32 <= miniL0abThreshold &&
        params.n32 * params.k32 <= miniL0abThreshold && params.m32 * params.n32 <= miniL0cThreshold &&
        !specialScenario) {
        coreStatus.batchDim = params.batch32;
        coreStatus.nDim = params.n32 <= MIN_MTE1_LOAD ? 1 : params.nMapped / MIN_MTE1_LOAD;
        coreStatus.mDim = params.m32 <= MIN_MTE1_LOAD ? 1 : params.mMapped / MIN_MTE1_LOAD;
        int32_t kDimCandidate[2] = {0}; // storage 2 factors of k around kDim
        GetTwoFactors(kDimCandidate, coreStatus.kDim, params.k32, coreNum);
        coreStatus.kDim = (params.k32 <= MIN_MTE1_LOAD || !splitKFlag) ?
            1 :
            (kDimCandidate[1] > 1 ? kDimCandidate[1] : kDimCandidate[0]);
        coreStatus.batch = 1;
        coreStatus.n = coreStatus.nDim == 1 ? params.n32 : MathUtil::CeilDivision(params.nMapped, coreStatus.nDim);
        coreStatus.m = coreStatus.mDim == 1 ? params.m32 : MathUtil::CeilDivision(params.mMapped, coreStatus.mDim);
        coreStatus.k = coreStatus.kDim == 1 ? params.k32 : MathUtil::CeilDivision(params.kMapped, coreStatus.kDim);
        params.nonFactorK = (coreStatus.kDim == 0) ? false : (params.k32 % coreStatus.kDim == 0 ? false : true);
        return true;
    }
    return false;
}
float MatmulTilingAlgorithm::CalculateBlockCycles(int32_t baseM, int32_t baseN, int32_t baseK) const
{
    const int32_t reduceBlockSize = C0_BYTE_SIZE * BITS_PER_BYTE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType);
    return static_cast<float>(baseM * baseN * baseK) / (C0_SIZE * C0_SIZE * reduceBlockSize);
}

float MatmulTilingAlgorithm::CalculateMemoryTraffic(int32_t baseM, int32_t baseN, int32_t baseK,
    float aMemoryRatio, float bMemoryRatio) const
{
    float aMatrixSize = aMemoryRatio * baseM * baseK * DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
    float bMatrixSize = bMemoryRatio * baseN * baseK * DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
    return aMatrixSize + bMatrixSize;
}

bool MatmulTilingAlgorithm::AlignSingleShape(bool needAlign, int32_t orgShape, int32_t factor, int32_t alignSize,
    int32_t &singleShape) const
{
    singleShape = MathUtil::CeilDivision(orgShape, factor);
    if (!needAlign || alignSize  == 0) {
        return true; // orgShape not align, don't need to adjust
    }
    if (factor <= 1) {
        return true;
    }
    int32_t maxSingleShape = MathUtil::CeilDivision(orgShape, factor - 1);
    int32_t alignSingleShape = MathUtil::Align(singleShape, alignSize);
    if (alignSingleShape >= maxSingleShape) {
        return false;
    }
    singleShape = alignSingleShape;
    return true;
}

ComputeBaseBlock MatmulTilingAlgorithm::GetMultiCoreBasicBlock(const MatmulRunParas& params) const
{
    (void)params;
    constexpr static int32_t l0c256KB = 262144;
    constexpr static int32_t basicSize128 = 128;
    constexpr static int32_t basicSize256 = 256;
    int32_t basicM = basicSize128;
    if (tilingIns_->bufferPool_.l0CSize == l0c256KB) {
        basicM = basicSize256;
    }
    int32_t basicN = basicSize256;
    int32_t aDtypeSize = DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) != 0 ?
        DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) : 1;
    int32_t basicK = basicSize128 * BITS_PER_BYTE / aDtypeSize;
    ComputeBaseBlock basicBlock {basicM, basicN, basicK};
    // SetFixSplit
    if (tilingIns_->baseM != -1) {
        basicBlock.baseM = tilingIns_->baseM;
    }
    if (tilingIns_->baseN != -1) {
        basicBlock.baseN = tilingIns_->baseN;
    }
    if (!tilingIns_->aType_.isTrans && !tilingIns_->bType_.isTrans) {
        return basicBlock;
    }
    if (tilingIns_->aType_.isTrans && tilingIns_->bType_.isTrans) {
        basicBlock.baseM = tilingIns_->baseM != -1 ? basicBlock.baseM : basicSize256;
        basicBlock.baseN = tilingIns_->baseN != -1 ? basicBlock.baseN : basicSize128;
        return basicBlock;
    }

    return basicBlock;
}

float MatmulTilingAlgorithm::CalcBaseBlockBandRatio(int32_t mDim, int32_t nDim, const ComputeBaseBlock &baseBlock) const
{
    float bandRatio = static_cast<float>((numOfBlock_ - mDim) * baseBlock.baseM + (numOfBlock_ - nDim) * baseBlock.baseN) /
        static_cast<float>((baseBlock.baseM + baseBlock.baseN) * numOfBlock_);
    return bandRatio;
}

void MatmulTilingAlgorithm::UpdateBaseBlock(const MatmulRunParas& params,
    const int32_t sm, const int32_t sn, ComputeBaseBlock &baseBlock) const
{
    constexpr int32_t basicSize128 = 128;
    constexpr int32_t basicSize256 = 256;
    constexpr int32_t basicSize384 = 384;
    constexpr int32_t basicSize32 = 32;
    constexpr int32_t basicSize64 = 64;
    baseBlock.baseK = min(baseBlock.baseK,
        MathUtil::Align(static_cast<int32_t>(params.oriShapeKa), BASIC_SIZE_32)); // 32 is more efficient than 16
    if (sm >= basicSize128) {
        baseBlock.baseM = basicSize128;
    } else {
        baseBlock.baseM = MathUtil::Align(sm, REDUCE_BLOCK_SIZE);
    }
    int32_t maxBaseN = basicSize256;
    if (baseBlock.baseK <= basicSize32 && baseBlock.baseM <= basicSize64) {
        maxBaseN = basicSize384;
    }
    if (sn >= maxBaseN) {
        baseBlock.baseN = maxBaseN;
    } else {
        baseBlock.baseN = MathUtil::Align(sn, REDUCE_BLOCK_SIZE);
    }
}

ComputeIntensity MatmulTilingAlgorithm::CalcComputeIntensity(const MatmulRunParas& params, const ComputeBaseBlock &baseBlock,
    const std::pair<int32_t, int32_t> &factor) const
{
    auto mFactor = factor.first;
    auto nFactor = factor.second;
    int32_t sm = 0;
    int32_t aAlignSize = DATA_COPY_ALIGN_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE;
    bool aNeedAlign = tilingIns_->aType_.isTrans && IsOrgShapeAlign(params.oriShapeM, aAlignSize);
    bool alignSuccA = AlignSingleShape(aNeedAlign, params.oriShapeM, mFactor, aAlignSize, sm);
    int32_t sn = 0;
    int32_t bAlignSize = DATA_COPY_ALIGN_SIZE / DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) * BITS_PER_BYTE;
    bool bNeedAlign = (!tilingIns_->bType_.isTrans) && IsOrgShapeAlign(params.oriShapeN, bAlignSize);
    bool alignSuccB = AlignSingleShape(bNeedAlign, params.oriShapeN, nFactor, bAlignSize, sn);
    auto shapeM = MathUtil::DivideIntoMainAndTail(sm, baseBlock.baseM);
    auto shapeN = MathUtil::DivideIntoMainAndTail(sn, baseBlock.baseN);
    
    int32_t memoryRatio = (alignSuccA && alignSuccB) ? 1 : 2;
    float bandRatio = CalcBaseBlockBandRatio(mFactor, nFactor, baseBlock);
    std::vector<BaseBlockIntensity> blocks = CalcTotalCycleMemory(shapeM, shapeN, baseBlock, memoryRatio);

    float totalCycles = 0;
    float totalMemory = 0;
    for (const auto& v : blocks) {
        totalCycles += v.computeCycle;
        totalMemory += v.memoryTraffic;
    }
    return {
        {mFactor, nFactor}, totalCycles, (totalMemory > 0) ? totalCycles / totalMemory : 0, bandRatio};
}

std::vector<BaseBlockIntensity> MatmulTilingAlgorithm::CalcTotalCycleMemory(const std::pair<int32_t, int32_t>& shapeM,
        const std::pair<int32_t, int32_t>& shapeN, const ComputeBaseBlock &baseBlock, const float memoryRatio,
        const MemoryRatios memoryRatios) const
{
    std::vector<BaseBlockIntensity> blocks;
    auto mainM = shapeM.first;
    auto tailM = shapeM.second;
    auto mainN = shapeN.first;
    auto tailN = shapeN.second;
    float aMemoryRatio = memoryRatios.aMemoryRatio;
    float bMemoryRatio = memoryRatios.bMemoryRatio;
    // Main Chunk
    if (mainM > 0 && mainN > 0) {
        int count = mainM * mainN;
        float cycles = CalculateBlockCycles(baseBlock.baseM, baseBlock.baseN, baseBlock.baseK) * count;
        float memory = memoryRatio *
            CalculateMemoryTraffic(baseBlock.baseM, baseBlock.baseN, baseBlock.baseK, aMemoryRatio, bMemoryRatio) * count;
        blocks.push_back({count, cycles, memory});
    }
    // N Tail Chunk
    if (mainM > 0 && tailN > 0) {
        float cycles = CalculateBlockCycles(baseBlock.baseM, tailN, baseBlock.baseK) * mainM;
        float memory = memoryRatio * CalculateMemoryTraffic(baseBlock.baseM, tailN, baseBlock.baseK,
            aMemoryRatio, bMemoryRatio) * mainM;
        blocks.push_back({mainM, cycles, memory});
    }
    // M Tail Chunk
    if (tailM > 0 && mainN > 0) {
        float cycles = CalculateBlockCycles(tailM, baseBlock.baseN, baseBlock.baseK) * mainN;
        float memory = memoryRatio * CalculateMemoryTraffic(tailM, baseBlock.baseN, baseBlock.baseK,
            aMemoryRatio, bMemoryRatio) * mainN;
        blocks.push_back({mainN, cycles, memory});
    }
    // M and N Tail Chunk
    if (tailM > 0 && tailN > 0) {
        float cycles = CalculateBlockCycles(tailM, tailN, baseBlock.baseK);
        float memory = memoryRatio * CalculateMemoryTraffic(tailM, tailN, baseBlock.baseK, aMemoryRatio, bMemoryRatio);
        blocks.push_back({1, cycles, memory});
    }
    return blocks;
}

ComputeIntensitySmallShape MatmulTilingAlgorithm::CalcComputeIntensitySmallShape(const MatmulRunParas& params,
    const std::pair<int32_t, int32_t> &factor, ComputeBaseBlock &baseBlock) const
{
    auto mFactor = factor.first;
    auto nFactor = factor.second;
    int32_t sm = 0;
    int32_t aAlignSize = DATA_COPY_ALIGN_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE;
    bool aNeedAlign = tilingIns_->aType_.isTrans && IsOrgShapeAlign(params.oriShapeM, aAlignSize, true);
    bool alignSuccA = AlignSingleShape(aNeedAlign, params.oriShapeM, mFactor, aAlignSize, sm);
    int32_t sn = 0;
    int32_t bAlignSize = DATA_COPY_ALIGN_SIZE / DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) * BITS_PER_BYTE;
    bool bNeedAlign = (!tilingIns_->bType_.isTrans) && IsOrgShapeAlign(params.oriShapeN, bAlignSize, true);
    bool alignSuccB = AlignSingleShape(bNeedAlign, params.oriShapeN, nFactor, bAlignSize, sn);
    (void)UpdateBaseBlock(params, sm, sn, baseBlock);
    auto shapeM = MathUtil::DivideIntoMainAndTail(sm, baseBlock.baseM);
    auto shapeN = MathUtil::DivideIntoMainAndTail(sn, baseBlock.baseN);
    float aMemoryRatio = alignSuccA ? 1 : MAX_BAND_WIDTH_RATIO;
    float bMemoryRatio = alignSuccB ? 1 : MAX_BAND_WIDTH_RATIO;
    // Update memoryRatio according to BAND_WIDTH_TAB
    if (!alignSuccA) {
        const auto iter = BAND_WIDTH_TAB.find(sm);
        if (iter != BAND_WIDTH_TAB.end()) {
            aMemoryRatio = iter->second;
        }
    }
    if (!alignSuccB) {
        const auto iter = BAND_WIDTH_TAB.find(sn);
        if (iter != BAND_WIDTH_TAB.end()) {
            bMemoryRatio = iter->second;
        }
    }
    int32_t bandRate = mFactor * nFactor;
    if ((sm < BASIC_SIZE_32 && sn < BASIC_SIZE_32) && bandRate > BAND_LIMIT_MAX_CORENUM) {
        bandRate = BAND_LIMIT_MAX_CORENUM;
    }
    bandRate = (bandRate == 0) ? 1 : bandRate; // check bandRate != 0
    float memoryRatio = static_cast<float>(mFactor * nFactor) / static_cast<float>(bandRate);
    float bandRatio = CalcBaseBlockBandRatio(mFactor, nFactor, baseBlock);
    MemoryRatios ratios(aMemoryRatio, bMemoryRatio);
    std::vector<BaseBlockIntensity> blocks = CalcTotalCycleMemory(shapeM, shapeN, baseBlock, memoryRatio, ratios);

    float totalCycles = 0;
    float totalMemory = 0;
    for (const auto& v : blocks) {
        totalCycles += v.computeCycle;
        totalMemory += v.memoryTraffic;
    }
    return {
        {mFactor, nFactor}, totalCycles, (totalMemory > 0) ? totalCycles / totalMemory : 0,
        bandRatio, totalMemory, baseBlock};
}

MultiCoreScenario MatmulTilingAlgorithm::GetMultiCoreScenario(const MatmulRunParas& params) const
{
    if (tilingIns_->socVersion != platform_ascendc::SocVersion::ASCEND910B &&
        tilingIns_->socVersion != platform_ascendc::SocVersion::ASCEND950) {
        return MultiCoreScenario::OTHERS;
    }
    if (tilingIns_->enableSplitK_ || tilingIns_->singleM != -1 || tilingIns_->singleN != -1) {
        return MultiCoreScenario::OTHERS;
    }
    constexpr int64_t mnLimit = 26214; // 128 * 256 * 0.8
    constexpr int64_t mLimit = 128;
    if (params.oriShapeM >= mLimit && params.oriShapeM * params.oriShapeN > mnLimit * numOfBlock_) {
        return MultiCoreScenario::SPLIT_MN;
    }
    if (params.oriShapeM * params.oriShapeN < mnLimit * numOfBlock_) {
        return MultiCoreScenario::SPLIT_SMALL_MN;
    }
    return MultiCoreScenario::OTHERS;
}

void MatmulTilingAlgorithm::UpdateStepK(const ComputeBaseBlock &baseBlock, int32_t &stepK) const
{
    if (stepK * baseBlock.baseK >= GetSingleK()) {
        return;
    }
    constexpr static int32_t baseBlockSize512 = 512;
    constexpr static int32_t baseBlockSize256 = 256;
    int32_t aTypeBitSize = DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType);
    if (stepK * baseBlock.baseK * aTypeBitSize / BITS_PER_BYTE > baseBlockSize512) {
        if ((stepK * baseBlock.baseK * aTypeBitSize / BITS_PER_BYTE % baseBlockSize512 != 0) &&
            (baseBlockSize512 % (baseBlock.baseK * aTypeBitSize / BITS_PER_BYTE) == 0)) {
            while (stepK * baseBlock.baseK * aTypeBitSize / BITS_PER_BYTE % baseBlockSize512 != 0 && stepK > 1) {
                stepK--;
            }
        }
    } else if (stepK * baseBlock.baseK * aTypeBitSize / BITS_PER_BYTE > baseBlockSize256) {
         if ((stepK * baseBlock.baseK * aTypeBitSize / BITS_PER_BYTE % baseBlockSize256 != 0) &&
            (baseBlockSize256 % (baseBlock.baseK * aTypeBitSize / BITS_PER_BYTE) == 0)) {
            while (stepK * baseBlock.baseK * aTypeBitSize / BITS_PER_BYTE % baseBlockSize256 != 0 && stepK > 1) {
                stepK--;
            }
        }       
    }
}

void MatmulTilingAlgorithm::CalcL1Tiling(const ComputeBaseBlock &baseBlock, int32_t &depthA1, int32_t &depthB1,
    int32_t &stepKa, int32_t &stepKb) const
{
    int32_t l1Size = tilingIns_->bufferPool_.l1Size;
    constexpr static int32_t reservedL1Size = 256; // l1 reserved 256B
    int32_t depthA1Size = (l1Size / DB_ON / baseBlock.baseM / baseBlock.baseK) * BITS_PER_BYTE /
        DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType);
    int32_t depthB1Size = ((l1Size + reservedL1Size) / DB_ON / baseBlock.baseN / baseBlock.baseK) * BITS_PER_BYTE /
        DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType);
    int32_t btSize = 0;
    if (tilingIns_->isBias) {
        btSize = baseBlock.baseN * DTYPE_BIT_TAB.at(tilingIns_->biasType_.dataType) / BITS_PER_BYTE;
    }
    int32_t baseAByteSize = GetABaseHeightAlign(baseBlock.baseM) * GetABaseWidthAlign(baseBlock.baseK) *
        DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
    int32_t baseBByteSize = GetBBaseHeightAlign(baseBlock.baseK) * GetBBaseWidthAlign(baseBlock.baseN) *
        DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
    if (depthA1Size * baseAByteSize + depthB1Size * baseBByteSize > l1Size - btSize) {
        if (GetABaseHeightAlign(baseBlock.baseM) <= GetBBaseWidthAlign(baseBlock.baseN)) {
            depthA1Size = depthA1Size / DB_ON;
        } else {
            depthB1Size = depthB1Size / DB_ON;
        }
    }
    int32_t l1Db = g_tempCfg.l1DB == DB_OFF ? DB_OFF : DB_ON;
    stepKa = depthA1Size / l1Db;
    stepKb = depthB1Size / l1Db;
    UpdateStepK(baseBlock, stepKa);
    UpdateStepK(baseBlock, stepKb);
    if (stepKa >= stepKb && stepKb != 0) {
        stepKa = stepKa / stepKb * stepKb;
    } else if (stepKa != 0) {
        stepKb = stepKb / stepKa * stepKa;
    }
    depthA1 = stepKa * l1Db;
    depthB1 = stepKb * l1Db;
}

L0StatusPack MatmulTilingAlgorithm::GetL0CoreStatus(const ComputeBaseBlock &baseBlock) const
{
    L0StatusPack l0Status;
    const int32_t reduceSize = C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE;
    l0Status.dbL0C = g_tempCfg.l0cDB;
    if (baseBlock.baseM * baseBlock.baseN > tilingIns_->bufferPool_.l0CSize / DB_ON) {
        l0Status.dbL0C = DB_OFF;
    }
    l0Status.dbL0A = DB_ON;
    l0Status.dbL0B = DB_ON;
    l0Status.mL0 = baseBlock.baseM / C0_SIZE;
    l0Status.kL0 = baseBlock.baseK / reduceSize;
    l0Status.nL0 = baseBlock.baseN / C0_SIZE;
    return l0Status;
}

L1StatusPack MatmulTilingAlgorithm::GetL1CoreStatus(const ComputeBaseBlock &baseBlock, int32_t depthA1, int32_t depthB1,
    int32_t stepKa, int32_t stepKb) const
{
    L1StatusPack l1Status;
    l1Status.mAL1 = 1;
    l1Status.nBL1 = 1;
    const int32_t reduceSize = C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE;
    l1Status.kAL1 = baseBlock.baseK / reduceSize * stepKa;
    l1Status.kBL1 = baseBlock.baseK / reduceSize * stepKb;
    l1Status.dbAL1 = depthA1 >= stepKa * DB_ON ? DB_ON : DB_OFF;
    l1Status.dbBL1 = depthB1 >= stepKb * DB_ON ? DB_ON : DB_OFF;
    return l1Status;
}

void MatmulTilingAlgorithm::UpdateShapeAndLayout() const
{
    tilingIns_->tiling_.set_M(tilingIns_->orgM);
    tilingIns_->tiling_.set_N(tilingIns_->orgN);
    tilingIns_->tiling_.set_Ka(tilingIns_->orgKa);
    tilingIns_->tiling_.set_Kb(tilingIns_->orgKb);
    tilingIns_->tiling_.set_batchM(tilingIns_->batchM);
    tilingIns_->tiling_.set_batchN(tilingIns_->batchN);
    tilingIns_->tiling_.set_singleBatchM(tilingIns_->singleBatchM);
    tilingIns_->tiling_.set_singleBatchN(tilingIns_->singleBatchN);

    tilingIns_->tiling_.set_ALayoutInfoB(tilingIns_->aLayoutInfoB);
    tilingIns_->tiling_.set_ALayoutInfoS(tilingIns_->aLayoutInfoS);
    tilingIns_->tiling_.set_ALayoutInfoN(tilingIns_->aLayoutInfoN);
    tilingIns_->tiling_.set_ALayoutInfoG(tilingIns_->aLayoutInfoG);
    tilingIns_->tiling_.set_ALayoutInfoD(tilingIns_->aLayoutInfoD);

    tilingIns_->tiling_.set_BLayoutInfoB(tilingIns_->bLayoutInfoB);
    tilingIns_->tiling_.set_BLayoutInfoS(tilingIns_->bLayoutInfoS);
    tilingIns_->tiling_.set_BLayoutInfoN(tilingIns_->bLayoutInfoN);
    tilingIns_->tiling_.set_BLayoutInfoG(tilingIns_->bLayoutInfoG);
    tilingIns_->tiling_.set_BLayoutInfoD(tilingIns_->bLayoutInfoD);

    tilingIns_->tiling_.set_CLayoutInfoB(tilingIns_->cLayoutInfoB);
    tilingIns_->tiling_.set_CLayoutInfoS1(tilingIns_->cLayoutInfoS1);
    tilingIns_->tiling_.set_CLayoutInfoN(tilingIns_->cLayoutInfoN);
    tilingIns_->tiling_.set_CLayoutInfoG(tilingIns_->cLayoutInfoG);
    tilingIns_->tiling_.set_CLayoutInfoS2(tilingIns_->cLayoutInfoS2);
    tilingIns_->tiling_.set_BatchNum(tilingIns_->batchNum);
    return;
}

void MatmulTilingAlgorithm::UpdateUsedSize() const
{
    int32_t transLength = 0;
    GetTransLength(transLength);
    int32_t a1LengthCache = 0;
    int32_t b1LengthCache = 0;
    SetDepthL1CacheUBParams(a1LengthCache, b1LengthCache);
    tilingIns_->tiling_.set_transLength(transLength); // a1 b1 c1 reuse in ub
    tilingIns_->tiling_.set_shareMode(0);
    int32_t l1Size = 0;
    int32_t l0cSize = 0;
    int32_t ubSize = 0;
    GetUsedSize(l1Size, l0cSize, ubSize, a1LengthCache, b1LengthCache);
    tilingIns_->tiling_.set_shareL1Size(l1Size);
    tilingIns_->tiling_.set_shareL0CSize(l0cSize);
    tilingIns_->tiling_.set_shareUbSize(ubSize);
}

int64_t MatmulTilingAlgorithm::AdjustOuterProductL0Factor(SingleCoreStatus& singleCoreStatus) const
{
    if (tilingIns_->scheduleType != ScheduleType::OUTER_PRODUCT) {
        return 0;
    }
    // check whether OUTER_PRODUCT is supported
    if ((tilingIns_->tiling_.get_baseK() < tilingIns_->tiling_.get_singleCoreK()) &&
        ((tilingIns_->mmConfigType == 1) || ((tilingIns_->mmConfigType == 0) &&
        (tilingIns_->batchNum != 0)))) {
        TILING_LOG_WARNING("MatmulApi Tiling : Unsupported scheduleType is OUTER_PRODUCT");
        return -1L;
    }
    int32_t newBaseM = singleCoreStatus.l0Status.mL0 * C0_SIZE;
    int32_t newBaseN = singleCoreStatus.l0Status.nL0 * C0_SIZE;
    int32_t newStepM = singleCoreStatus.l1Status.mAL1;
    int32_t newStepN = singleCoreStatus.l1Status.nBL1;
    // when scheduleType is OUTER_PRODUCT, each iteration computes 2 * basicBlock size of data
    bool isL0CFullUsed = (newBaseM * newBaseN * NUM_TWO * FP32_BYTES) >
        static_cast<int32_t>(tilingIns_->bufferPool_.l0CSize) ? true : false;
    if (isL0CFullUsed && (tilingIns_->tiling_.get_iterateOrder() == 0)) {
        // when scheduleType is OUTER_PRODUCT and iterateOrder is ORDER_M, N db in L0
        newBaseN = MathUtil::Align(newBaseN / NUM_TWO, C0_SIZE);
        newStepN *= NUM_TWO;
    } else if (isL0CFullUsed && (tilingIns_->tiling_.get_iterateOrder() == 1)) {
        // when scheduleType is OUTER_PRODUCT and iterateOrder is ORDER_N, M db in L0
        newBaseM = MathUtil::Align(newBaseM / NUM_TWO, C0_SIZE);
        newStepM *= NUM_TWO;
    }
    tilingIns_->tiling_.set_baseM(newBaseM);
    tilingIns_->tiling_.set_baseN(newBaseN);
    singleCoreStatus.l1Status.mAL1 = newStepM;
    singleCoreStatus.l1Status.nBL1 = newStepN;
    return 0;
}

void MatmulTilingAlgorithm::AdjustFloatL1Factor(const SingleCoreStatus& singleCoreStatus) const
{
    if (DTYPE_BYTE_TAB.at(tilingIns_->bType_.dataType) == DTYPE_BYTE_TAB.at(DataType::DT_FLOAT)) {
        if (tilingIns_->tiling_.get_baseK() == DT_FLOAT_INVALID_BASEK) {
            tilingIns_->tiling_.set_stepKb(1);
            tilingIns_->tiling_.set_depthB1(singleCoreStatus.l1Status.nBL1 * singleCoreStatus.l1Status.dbBL1);
        }
    }
}

void MatmulTilingAlgorithm::SetBaseMNK(const SingleCoreStatus& singleCoreStatus) const
{
    int32_t baseM = singleCoreStatus.l0Status.mL0 * C0_SIZE;
    if (tilingIns_->madType_ == MatrixMadType::MXMODE) {
        bool isAUbNdTrans = tilingIns_->aType_.pos == TPosition::VECOUT && 
            tilingIns_->aType_.type == CubeFormat::ND && tilingIns_->aType_.isTrans;
        int32_t widthAlignSize = INT8_ALIGN_SIZE;
        if (tilingIns_->aType_.dataType == DataType::DT_FLOAT4_E2M1 ||
            tilingIns_->aType_.dataType == DataType::DT_FLOAT4_E1M2) {
            widthAlignSize = INT4_ALIGN_SIZE;
        }
        if (isAUbNdTrans) {
            baseM = MathUtil::CeilDivision(baseM, widthAlignSize) * widthAlignSize;
        }
    }
    tilingIns_->tiling_.set_baseM(baseM);
    tilingIns_->tiling_.set_baseN(singleCoreStatus.l0Status.nL0 * C0_SIZE);
    const int32_t reduceSize = C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE;
    tilingIns_->tiling_.set_baseK(singleCoreStatus.l0Status.kL0 * reduceSize);
}

int64_t MatmulTilingAlgorithm::UpdateTiling(const MatmulRunParas& param, const CoreStatusPack &coreStatus, SingleCoreStatus& singleCoreStatus) const
{
    int32_t coreUse = enableSingleShape_ ? tilingIns_->blockDim : coreStatus.batchDim * coreStatus.mDim * coreStatus.kDim * coreStatus.nDim;
    int32_t singleCoreM;
    int32_t singleCoreN;
    int32_t singleCoreK;
    GetSingleShape(coreStatus, param, singleCoreM, singleCoreN, singleCoreK);
    if (!CheckSingleShape(singleCoreM, singleCoreN, singleCoreK)) {
        return -1L;
    }
    tilingIns_->tiling_.set_usedCoreNum(coreUse);
    tilingIns_->tiling_.set_singleCoreM(singleCoreM);
    tilingIns_->tiling_.set_singleCoreN(singleCoreN);
    tilingIns_->tiling_.set_singleCoreK(singleCoreK);
    UpdateShapeAndLayout();
    SetBaseMNK(singleCoreStatus);
    tilingIns_->tiling_.set_iterateOrder(GetIteratorOrder(singleCoreStatus, singleCoreM, singleCoreN, singleCoreK));
    // check whether OUTER_PRODUCT is supported
    if (AdjustOuterProductL0Factor(singleCoreStatus) != 0) {
        return -1L;
    }
    tilingIns_->baseM = tilingIns_->tiling_.get_baseM();
    tilingIns_->baseN = tilingIns_->tiling_.get_baseN();
    tilingIns_->baseK = tilingIns_->tiling_.get_baseK();
    AdjustMxL1Factors(singleCoreStatus);
    if (!AdjustNBuffer33L1Factors(coreStatus, singleCoreStatus)) {
        return -1L;
    }
    tilingIns_->tiling_.set_depthA1(MathUtil::CeilDivision(singleCoreStatus.l1Status.kAL1,
        singleCoreStatus.l0Status.kL0) * singleCoreStatus.l1Status.mAL1 * singleCoreStatus.l1Status.dbAL1);
    tilingIns_->tiling_.set_depthB1(UpdateDepthB1(singleCoreStatus));
    // if decrease depthB1, nBL1 must decrease to ensure nBL1 is less then depthB1
    singleCoreStatus.l1Status.nBL1 = min(singleCoreStatus.l1Status.nBL1, tilingIns_->tiling_.get_depthB1());
    tilingIns_->tiling_.set_stepM(singleCoreStatus.l1Status.mAL1);
    tilingIns_->tiling_.set_stepN(singleCoreStatus.l1Status.nBL1);
    tilingIns_->tiling_.set_stepKa(MathUtil::CeilDivision(singleCoreStatus.l1Status.kAL1, singleCoreStatus.l0Status.kL0));
    tilingIns_->tiling_.set_stepKb(MathUtil::CeilDivision(singleCoreStatus.l1Status.kBL1, singleCoreStatus.l0Status.kL0));
    int32_t mxTypePara = 0;
    // determine whether the scenario is MX
    if (tilingIns_->madType_ == MatrixMadType::MXMODE) {
        GetMxScaleFactor(singleCoreStatus, mxTypePara);
    }
    tilingIns_->tiling_.set_mxTypePara(mxTypePara);

    AdjustFloatL1Factor(singleCoreStatus);
    tilingIns_->tiling_.set_isBias(tilingIns_->isBias ? 1 : 0);
    tilingIns_->tiling_.set_dbL0A(singleCoreStatus.l0Status.dbL0A);
    tilingIns_->tiling_.set_dbL0B(singleCoreStatus.l0Status.dbL0B);
    tilingIns_->tiling_.set_dbL0C(singleCoreStatus.l0Status.dbL0C);
    UpdateUsedSize();
    return 0;
}

bool MatmulTilingAlgorithm::DoMultiCoreSplitMNTiling(const MatmulRunParas& params, CoreStatusPack& coreStatus,
    DimCalculator& dimCalRes)
{
    auto multiCoreScenario = GetMultiCoreScenario(params);
    if (multiCoreScenario != MultiCoreScenario::SPLIT_MN && multiCoreScenario != MultiCoreScenario::SPLIT_SMALL_MN &&
        tilingIns_->scheduleType != ScheduleType::N_BUFFER_33) {
        return false;
    }
    ComputeBaseBlock baseBlock = GetMultiCoreBasicBlock(params); // calc basic block
    if (tilingIns_->scheduleType == ScheduleType::N_BUFFER_33) {
        if (!CalcNBuffer33Dims(params, baseBlock, coreStatus)) {
            return false;
        }
    } else if (multiCoreScenario == MultiCoreScenario::SPLIT_MN) {
        TILING_LOG_DEBUG("Multi-core scenario is SPLIT_MN.");
        CalcMultiCoreDims(params, baseBlock, coreStatus, dimCalRes);
    } else {
        TILING_LOG_DEBUG("Multi-core scenario is SPLIT_SMALL_MN.");
        CalcMultiCoreDimsSmallShape(params, baseBlock, coreStatus, dimCalRes);
    }

    SingleCoreStatus singleCoreStatus;
    singleCoreStatus.l0Status = GetL0CoreStatus(baseBlock);
    AdjustSparseL0Factors(singleCoreStatus);
    AdjustMxL0Factors(singleCoreStatus);
    int32_t depthA1;
    int32_t depthB1;
    int32_t stepKa;
    int32_t stepKb;
    CalcL1Tiling(baseBlock, depthA1, depthB1, stepKa, stepKb);
    singleCoreStatus.l1Status = GetL1CoreStatus(baseBlock, depthA1, depthB1, stepKa, stepKb);
    if (UpdateTiling(params, coreStatus, singleCoreStatus) == -1L) {
        return false;
    }
    return true;
}

bool MatmulTilingAlgorithm::NeedOutputAlign(int32_t m, int32_t n, int32_t k) const
{
    int32_t aTypeSize = DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType);
    int32_t bTypeSize = DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType);
    int32_t cTypeSize = DTYPE_BIT_TAB.at(tilingIns_->cType_.dataType);
    constexpr static int32_t outputRatio = 2;
    bool needAlign = static_cast<int64_t>(n * m) * static_cast<int64_t>(outputRatio * cTypeSize) >
        static_cast<int64_t>(n * k* aTypeSize) + static_cast<int64_t>(m * k * bTypeSize);
    return needAlign;
}

bool MatmulTilingAlgorithm::CalcNBuffer33Dims(const MatmulRunParas& params, const ComputeBaseBlock &baseBlock,
    CoreStatusPack& coreStatus) const
{
    coreStatus.batchDim = 1;
    coreStatus.mDim = MathUtil::CeilDivision(GetSingleM(), baseBlock.baseM * N_BUFFER_33_FACTOR);
    if (tilingIns_->enableSplitK_) {
        coreStatus.kDim = MathUtil::CeilDivision(GetSingleK(), baseBlock.baseK * N_BUFFER_33_FACTOR);
    } else {
        coreStatus.kDim = 1;
        if (MathUtil::CeilDivision(GetSingleK(), baseBlock.baseK) > N_BUFFER_33_FACTOR) {
            TILING_LOG_WARNING("MatmulApi Tiling : SingleCoreK %d and baseK %d does not satisfy NBuffer33 requirements. "
                "Suggest use EnableMultiCoreSplitK to turn on multi core K split.", GetSingleK(), baseBlock.baseK);
            return false;
        }
    }
    if (coreStatus.mDim * coreStatus.kDim > numOfBlock_) {
        TILING_LOG_WARNING("MatmulApi Tiling : M %d (baseM %d) or K %d (baseK %d) is too large to find a valid NBuffer33 single core "
            "shape within %d cores. Remind to slice M or K in test code.", GetSingleM(), baseBlock.baseM,
            GetSingleK(), baseBlock.baseK, numOfBlock_);
    }

    std::vector<std::pair<int32_t, int32_t>> dimPairs;
    int32_t nDim = 1;
    dimPairs.push_back({coreStatus.mDim, nDim});
    int32_t nDimMax = min(GetSingleN() / baseBlock.baseN, numOfBlock_ / (coreStatus.mDim * coreStatus.kDim));
    while (nDim <= nDimMax) {
        nDim++;
        dimPairs.push_back({coreStatus.mDim, nDim});
    }
    std::vector<ComputeIntensity> results;
    for (const auto &factor : dimPairs) {
        results.push_back(CalcComputeIntensity(params, baseBlock, factor));
    }
    std::sort(results.begin(), results.end());
    for (const auto &res : results) {
        TILING_LOG_DEBUG("intent:%f, cycle: %f, band: %f, mDim: %d, nDim: %d\n",
            res.avgIntensity, res.computeCycle, res.bandRatio, res.dimFactor.first, res.dimFactor.second);
    }
    coreStatus.nDim = results[0].dimFactor.second;
    return true;
}

void MatmulTilingAlgorithm::CalcMultiCoreDims(const MatmulRunParas& params, const ComputeBaseBlock &baseBlock,
    CoreStatusPack& coreStatus, DimCalculator& dimCalRes)
{
    auto factors = MathUtil::GetFactorPairs(numOfBlock_);
    std::vector<ComputeIntensity> results;
    for (const auto& factor : factors) {
        results.push_back(CalcComputeIntensity(params, baseBlock, factor));
    }
    // 排序结果
    std::sort(results.begin(), results.end());
    for (auto v : results) {
        TILING_LOG_DEBUG("intent:%f, cycle: %f, band: %f, mDim: %d, nDim: %d\n",
            v.avgIntensity, v.computeCycle, v.bandRatio, v.dimFactor.first, v.dimFactor.second);
    }
    coreStatus.batchDim = 1;
    dimCalRes.nDimFactor = results[0].dimFactor.second;
    dimCalRes.mDimFactor = results[0].dimFactor.first;
    dimCalRes.kDimFactor = 1;
    coreStatus.mDim = results[0].dimFactor.first;
    coreStatus.nDim = results[0].dimFactor.second;
    coreStatus.kDim = 1;
    (void)CalcMultiCoreDimsPost(params, coreStatus, dimCalRes);
    return;
}

void MatmulTilingAlgorithm::CalcMultiCoreDimsSmallShape(const MatmulRunParas& params, ComputeBaseBlock &baseBlock,
    CoreStatusPack& coreStatus, DimCalculator& dimCalRes)
{
    int32_t basicSize128 =128;
    int32_t basicSize256 =256;
    //if a certain axis can be fully divided, divided it fully
    if (params.oriShapeM <= static_cast<int64_t>(basicSize128) && 
        params.oriShapeN >= static_cast<int64_t>(basicSize128 * numOfBlock_)) {
        dimCalRes.mDimFactor = 1;
        dimCalRes.nDimFactor = numOfBlock_;
        coreStatus.mDim = 1;
        coreStatus.nDim = numOfBlock_;
        (void)UpdateBaseBlock(params, static_cast<int32_t>(params.oriShapeM),
            static_cast<int32_t>(params.oriShapeN) / numOfBlock_, baseBlock);
    } else if (params.oriShapeN <= static_cast<int64_t>(basicSize256) &&
                params.oriShapeM >= static_cast<int64_t>(basicSize128 * numOfBlock_)) {
        dimCalRes.mDimFactor = numOfBlock_;
        dimCalRes.nDimFactor = 1;
        coreStatus.mDim = numOfBlock_;
        coreStatus.nDim = 1;
        (void)UpdateBaseBlock(params, static_cast<int32_t>(params.oriShapeM) / numOfBlock_,
            static_cast<int32_t>(params.oriShapeN), baseBlock);
    } else {
        std::vector<ComputeIntensitySmallShape> results;
        for (int32_t i = numOfBlock_; i >= 1; i--) {
            auto factors = MathUtil::GetFactorPairs(i);
            for (const auto& factor : factors) {
                results.push_back(CalcComputeIntensitySmallShape(params, factor, baseBlock));
            }
        }
        std::sort(results.begin(), results.end());
        for (auto v : results) {
            TILING_LOG_DEBUG("memory:%f, intent:%f, cycle: %f, band: %f, mDim: %d, nDim: %d\n",
                v.memoryTraffic, v.avgIntensity, v.computeCycle, v.bandRatio, v.dimFactor.first, v.dimFactor.second);
        }
        coreStatus.batchDim = 1;
        dimCalRes.nDimFactor = results[0].dimFactor.second;
        dimCalRes.mDimFactor = results[0].dimFactor.first;
        dimCalRes.kDimFactor = 1;
        coreStatus.mDim = results[0].dimFactor.first;
        coreStatus.nDim = results[0].dimFactor.second;
        coreStatus.kDim = 1;
        
        baseBlock.baseM = results[0].baseBlock.baseM;
        baseBlock.baseN = results[0].baseBlock.baseN;
        baseBlock.baseK = results[0].baseBlock.baseK;   
    }

    int32_t tmpSize = L0_SIZE / DB_ON * BITS_PER_BYTE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType);
    baseBlock.baseK = min(tmpSize / baseBlock.baseM, tmpSize / baseBlock.baseN);
    baseBlock.baseK = min(MathUtil::AlignDown(baseBlock.baseK, GetC0Size()),
        static_cast<int32_t>(MathUtil::Align(params.oriShapeKa, static_cast<int64_t>(GetC0Size()))));
    (void)CalcMultiCoreDimsPost(params, coreStatus, dimCalRes);
    return;
}

void MatmulTilingAlgorithm::CalcMultiCoreDimsPost(const MatmulRunParas& params, CoreStatusPack& coreStatus,
    DimCalculator& dimCalRes)
{
    const int32_t n = MathUtil::FindBestSingleCore(params.n32, params.nMapped, dimCalRes.nDimFactor, false);
    const int32_t m = MathUtil::FindBestSingleCore(params.m32, params.mMapped, dimCalRes.mDimFactor, false);
    int32_t aAlignSize = DATA_COPY_ALIGN_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE;
    int32_t bAlignSize = DATA_COPY_ALIGN_SIZE / DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) * BITS_PER_BYTE;
    bool needOutputAlign = NeedOutputAlign(m, n, GetSingleK());
    (void)AlignSingleShape((!tilingIns_->bType_.isTrans || needOutputAlign), n, coreStatus.nDim, bAlignSize, coreStatus.n);
    (void)AlignSingleShape(tilingIns_->aType_.isTrans, m, coreStatus.mDim, aAlignSize, coreStatus.m);
    dimCalRes.kNum = params.k32 / coreStatus.kDim * C0_SIZE * REDUCE_BLOCK_SIZE; // contain k * 16
    dimCalRes.kBytes = dimCalRes.kNum * INPUTDTYPE_BYTES;                 // contain k * 16 * 2
    coreStatus.batch = params.batch32;
    coreStatus.k = params.k32 / coreStatus.kDim;
    TILING_LOG_DEBUG("CalcMultiCoreDims, coreStatus m: %d n: %d k: %d.", coreStatus.m, coreStatus.n, coreStatus.k);
    // load size of A matrix is batch * m
    // load size of B matrix is n
    DimFactor dimFactor(1, dimCalRes.mDimFactor, dimCalRes.kDimFactor, dimCalRes.nDimFactor);
    GetDimsHelper(dimFactor, coreStatus, dimCalRes, params);
    return;
}

void MatmulTilingAlgorithm::UpdateMultiCore(const std::string& opType, const MatmulRunParas& params,
    CoreStatusPack& coreStatus, const DimCalculator& dimCalRes) const
{
    (void)(opType);
    // Due to the modification of data amount in single-core, the number of multi-core needs to be updated.
    coreStatus.batchDim = min(MathUtil::CeilDivision(params.batch32, coreStatus.batch), numOfBlock_);
    coreStatus.nDim = min(MathUtil::CeilDivision(params.n32, coreStatus.n), numOfBlock_);
    coreStatus.mDim = min(MathUtil::CeilDivision(params.m32, coreStatus.m), numOfBlock_);

    if (tilingIns_->enableSplitK_) {
        coreStatus.kDim = min(MathUtil::CeilDivision(params.k32, coreStatus.k), numOfBlock_);
    } else {
        coreStatus.kDim = dimCalRes.kDimFactor;
    }
    UpdateBufferSize(tilingIns_->bType_.pos == TPosition::TSCM ? TilingPolicy::FIXED_B_TSCM : TilingPolicy::NO_POLICY,
        coreStatus);
}

void MatmulTilingAlgorithm::UpdateBufferSize(const TilingPolicy policy, const CoreStatusPack& coreStatus) const
{
    if (policy == TilingPolicy::NO_POLICY) {
        return;
    } else if (policy == TilingPolicy::FIXED_B_TSCM) {
        const int32_t bMatrixSize =
            MathUtil::Align(coreStatus.n, MathUtil::CeilDivision(tilingIns_->alignSingleN, C0_SIZE)) * coreStatus.k *
            C0_SIZE * C0_BYTE_SIZE * 2;
        tilingIns_->bufferPool_.l1Size -= bMatrixSize;
    } else if (policy == TilingPolicy::FIXED_A_TSCM) {
        const int32_t aMatrixSize = coreStatus.m * coreStatus.k * C0_SIZE * C0_BYTE_SIZE * 2;
        tilingIns_->bufferPool_.l1Size -= aMatrixSize;
    } else {
        return;
    }
}

bool MatmulTilingAlgorithm::IsInvalidFactor(int32_t factor) const
{
    return factor > numOfBlock_ || factor <= 0;
}

void MatmulTilingAlgorithm::AddOptimalFactors(const std::string& opType, const MatmulRunParas& params,
    DimCalculator& dimCalRes) const
{
    (void)(opType);
    const int32_t coreNum = numOfBlock_;
    // A/B fullLoad or A fullLoad + B Kdim fullLoad or B fullLoad + A Kdim fullLoad(1/2/4)
    const int32_t mnCore = MathUtil::CeilDivision(coreNum, params.batch32);
    if (mnCore > 1) {
        const float optPoint = static_cast<float>(sqrt((params.m32 + 0.0f) / params.n32 * mnCore));
        const int32_t mdim = static_cast<int32_t>(ceil(optPoint));
        const int32_t ndim = static_cast<int32_t>(ceil(mnCore / optPoint));
        MathUtil::AddFactor(dimCalRes.mDimFactors, mdim);
        MathUtil::AddFactor(dimCalRes.mDimFactors, ndim == 0 ? 1 : mnCore / ndim);
        MathUtil::AddFactor(dimCalRes.nDimFactors, ndim);
        MathUtil::AddFactor(dimCalRes.nDimFactors, mdim == 0 ? 1 : mnCore / mdim);
    }
}

void MatmulTilingAlgorithm::GenDimsMapFactors(const std::string& opType, MatmulRunParas& params,
    DimCalculator& dimCalRes) const
{
    const int32_t coreNum = numOfBlock_;
    dimCalRes.batchDimFactors.reserve(coreNum);
    dimCalRes.mDimFactors.reserve(coreNum);
    dimCalRes.nDimFactors.reserve(coreNum);
    dimCalRes.kDimFactors.reserve(coreNum);
    MathUtil::GetBlockFactors(dimCalRes.batchDimFactors, params.batch32, params.batchMapped, coreNum,
        min(coreNum, params.batch32));
    MathUtil::GetBlockFactors(dimCalRes.mDimFactors, params.m32, params.mMapped, coreNum, min(coreNum, params.m32));
    MathUtil::GetBlockFactors(dimCalRes.nDimFactors, params.n32, params.nMapped, coreNum, min(coreNum, params.n32));
    // first get kDim candidate
    if (!tilingIns_->enableSplitK_) {
        dimCalRes.kDimFactors.push_back(1);
        params.kMapped = params.k32;
    } else {
        MathUtil::GetBlockFactors(dimCalRes.kDimFactors, params.k32, params.kMapped, coreNum, coreNum);
    }
    AddOptimalFactors(opType, params, dimCalRes);
}

void MatmulTilingAlgorithm::GetDims(const std::string& opType, MatmulRunParas& params, CoreStatusPack& coreStatus,
    DimCalculator& dimCalRes)
{
    // get batchDim, kDim, mDim and nDim for single core
    // support multi cores slicing along kDim
    // single core batchDim, mDim, nDim, kDim is a factor of input batch, m, n, k
    // multi-core strategy for mini shape's is different from other situations and requires preprocess
    if (PreProcessMiniShape(opType, coreStatus, params, numOfBlock_, tilingIns_->enableSplitK_)) {
        // Due to the modification of data amount in single-core, the number of multi-core needs to be updated.
        coreStatus.batchDim = MathUtil::CeilDivision(params.batch32, coreStatus.batch);
        coreStatus.nDim = MathUtil::CeilDivision(params.n32, coreStatus.n);
        coreStatus.mDim = MathUtil::CeilDivision(params.m32, coreStatus.m);
        coreStatus.kDim = MathUtil::CeilDivision(params.k32, coreStatus.k);
        UpdateBufferSize(tilingIns_->bType_.pos == TPosition::TSCM ? TilingPolicy::FIXED_B_TSCM :
                                                                     TilingPolicy::NO_POLICY,
            coreStatus);
        splitCoreFlag_ = true;
        return;
    }
    GenDimsMapFactors(opType, params, dimCalRes);
    for (const int32_t bFactor : dimCalRes.batchDimFactors) {
        for (const int32_t nFactor : dimCalRes.nDimFactors) {
            if (IsInvalidFactor(bFactor * nFactor)) {
                continue;
            }
            for (const int32_t mFactor : dimCalRes.mDimFactors) {
                if (IsInvalidFactor(bFactor * nFactor * mFactor)) {
                    continue;
                }
                for (const int32_t kFactor : dimCalRes.kDimFactors) {
                    if (IsInvalidFactor(bFactor * nFactor * mFactor * kFactor)) {
                        continue;
                    }
                    DimFactor dimFactor(bFactor, mFactor, kFactor, nFactor);
                    GetDimsHelper(dimFactor, coreStatus, dimCalRes, params);
                }
            }
        }
    }

    coreStatus.batch = MathUtil::CeilDivision(params.batch32, dimCalRes.batchDimFactor);
    coreStatus.n = MathUtil::CeilDivision(params.n32, dimCalRes.nDimFactor);
    coreStatus.m = MathUtil::CeilDivision(params.m32, dimCalRes.mDimFactor);
    coreStatus.k = MathUtil::CeilDivision(params.k32, dimCalRes.kDimFactor);
    if (g_tempCfg.factorSplit) {
        const int32_t n = MathUtil::FindBestSingleCore(params.n32, params.nMapped, dimCalRes.nDimFactor, false);
        const int32_t m = MathUtil::FindBestSingleCore(params.m32, params.mMapped, dimCalRes.mDimFactor, false);
        const int32_t k = MathUtil::FindBestSingleCore(params.k32, params.kMapped, dimCalRes.kDimFactor, true);
        const int32_t needCoreNum = static_cast<int32_t>(MathUtil::CeilDivision(params.batch32, coreStatus.batch) *
                            MathUtil::CeilDivision(params.n32, n) *
                            MathUtil::CeilDivision(params.m32, m) *
                            MathUtil::CeilDivision(params.k32, k));
        if (IsInvalidFactor(needCoreNum) == false) {
            coreStatus.n = n;
            coreStatus.m = m;
            coreStatus.k = k;
        }
    }

    params.nonFactorK = params.k32 == params.kMapped ? false : true;
    UpdateMultiCore(opType, params, coreStatus, dimCalRes);
}

void MatmulTilingAlgorithm::NonFactorMap(const std::string& opType, MatmulRunParas& param,
    DimCalculator& dimCalRes) const
{
    (void)(opType);
    param.batchMapped = param.batch32;
    param.mMapped = param.m32;
    param.kMapped = param.k32;
    param.nMapped = param.n32;
    // Split k will introduce atomic_add which can't be used with shift_inwards.
    // Thus in split k mode, batch/m/n/ can't use non-factorial segmentation.
    if (tilingIns_->enableSplitK_) {
        // it is only necessary to consider the non-factor splitting of k when splitKFlag is true
        int32_t kFactorLess64Cnt = 0;
        int32_t kFactorLess1024Cnt = 0;
        MathUtil::GetFactorCnt(param.k32, kFactorLess64Cnt, 1, L0_FACTOR_LIMIT);
        MathUtil::GetFactorCnt(param.k32, kFactorLess1024Cnt, L0_FACTOR_LIMIT + 1, L1_FACTOR_LIMIT);
        if ((param.k32 > L0_FACTOR_LIMIT && kFactorLess64Cnt <= L0_FACTOR_NUM_LIMIT) ||
            (param.k32 > L1_FACTOR_LIMIT && kFactorLess64Cnt + kFactorLess1024Cnt <= L1_FACTOR_NUM_LIMIT)) {
            // Non-factors of the k dimension use a down-aligned number of powers of 2
            param.kMapped = MathUtil::MapShape(param.k32, false);
        }
    } else {
        MathUtil::GetFactorCnt(param.batch32, dimCalRes.batchFactorCnt, 1, numOfBlock_);
        if (param.batch32 > 1 && dimCalRes.batchFactorCnt <= L0_FACTOR_NUM_LIMIT) {
            param.batchMapped = MathUtil::MapShape(param.batch32);
        }
        param.mMapped = MathUtil::MapShape(param.m32);
        param.nMapped = MathUtil::MapShape(param.n32);
    }
}

void MatmulTilingAlgorithm::FillParam(MatmulRunParas& param)
{
    param.oriShapeM = tilingIns_->orgM;
    param.oriShapeN = tilingIns_->orgN;
    param.oriShapeKa = tilingIns_->orgKa;
    param.oriShapeKb = tilingIns_->orgKb;
    int32_t realM = 1;
    int32_t realN = 1;
    int32_t realK = 1;

    if (tilingIns_->singleCoreM != -1 || tilingIns_->singleCoreK != -1 || tilingIns_->singleCoreN != -1) {
        realM = tilingIns_->singleCoreM != -1 ? tilingIns_->singleCoreM : tilingIns_->singleM;
        realK = tilingIns_->singleCoreK != -1 ? tilingIns_->singleCoreK : tilingIns_->singleK;
        realN = tilingIns_->singleCoreN != -1 ? tilingIns_->singleCoreN : tilingIns_->singleN;
        enableSingleShape_ = true;
        numOfBlock_ = 1;
    } else {
        realM = GetSingleM();
        realK = GetSingleK();
        realN = GetSingleN();
        enableSingleShape_ = false;
        numOfBlock_ = tilingIns_->blockDim;
    }

    const int32_t reduceBlockSize = C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE;
    param.k32 = MathUtil::CeilDivision(realK, reduceBlockSize);
    param.m32 = MathUtil::CeilDivision(realM, C0_SIZE);
    param.n32 = MathUtil::CeilDivision(realN, C0_SIZE);
    param.mMapped = MathUtil::MapShape(param.m32, true);
    param.kMapped = MathUtil::MapShape(param.k32, true);
    param.nMapped = MathUtil::MapShape(param.n32, true);
}

bool MatmulTilingAlgorithm::CheckFinalParams(const CoreStatusPack& coreStatus) const
{
    (void)coreStatus;
    const int32_t stepM = tilingIns_->tiling_.get_stepM();
    const int32_t stepN = tilingIns_->tiling_.get_stepN();
    const int32_t depthA1 = tilingIns_->tiling_.get_depthA1();
    const int32_t depthB1 = tilingIns_->tiling_.get_depthB1();

    const int32_t l1Size = tilingIns_->tiling_.get_shareL1Size();
    const int32_t l0CSize = tilingIns_->tiling_.get_shareL0CSize();
    const int32_t uBSize = tilingIns_->tiling_.get_shareUbSize();

    if (stepM == 0 || stepN == 0 || depthA1 == 0 || depthB1 == 0) {
        TILING_LOG_WARNING("MatmulApi Tiling : stepM/N  depthA1/B1 should greater then zeros");
        return false;
    }

    if (stepM > depthA1 || stepN > depthB1) {
        TILING_LOG_WARNING("MatmulApi Tiling : stepM/N should less then depthA1/B1");
        return false;
    }

    if (l1Size > tilingIns_->bufferPool_.l1Size || l0CSize > tilingIns_->bufferPool_.l0CSize ||
        uBSize > tilingIns_->bufferPool_.ubSize) {
        TILING_LOG_WARNING("MatmulApi Tiling : L1/L0C/UB used size should less then L1Size/L0CSize/UbSize");
        return false;
    }

    int dateDtypeSize = DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType);
    int32_t biasL1Size = tilingIns_->isBias ?
        tilingIns_->tiling_.get_singleCoreN() * tilingIns_->tiling_.get_BatchNum() * dateDtypeSize / BITS_PER_BYTE : 0;
    if (!tilingIns_->isBMNKBmm && tilingIns_->tiling_.get_BatchNum() > 0 &&
        ((tilingIns_->tiling_.get_singleCoreM() * tilingIns_->tiling_.get_singleCoreK() +
        tilingIns_->tiling_.get_singleCoreN() * tilingIns_->tiling_.get_singleCoreK()) *
        tilingIns_->tiling_.get_BatchNum() * dateDtypeSize / BITS_PER_BYTE + biasL1Size >
        tilingIns_->bufferPool_.l1Size)) {
        TILING_LOG_WARNING("MatmulApi Tiling : a/b matrix size of batch mm should less then L1Size");
        return false;
    }

    return true;
}

void MatmulTilingAlgorithm::CheckL0DB(SingleCoreStatus& singleCoreStatus, const int32_t baseK) const
{
    int32_t baseM = singleCoreStatus.l0Status.mL0 * C0_SIZE;
    int32_t baseN = singleCoreStatus.l0Status.nL0 * C0_SIZE;
    if (tilingIns_->aType_.type == CubeFormat::ND && tilingIns_->aType_.isTrans &&
        tilingIns_->aType_.scalePos == TPosition::TSCM) {
        baseM = MathUtil::Align(singleCoreStatus.l0Status.mL0, L0_FACTOR_NUM_LIMIT) * C0_SIZE;
    }
    if (tilingIns_->bType_.type == CubeFormat::ND && !tilingIns_->bType_.isTrans &&
        tilingIns_->bType_.scalePos == TPosition::TSCM) {
        baseN = MathUtil::Align(singleCoreStatus.l0Status.nL0, L0_FACTOR_NUM_LIMIT) * C0_SIZE;
    }
    if (!tilingIns_->bType_.isTrans && (tilingIns_->bType_.dataType == DataType::DT_FLOAT4_E2M1 ||
            tilingIns_->bType_.dataType == DataType::DT_FLOAT4_E1M2)) {
        baseN = MathUtil::Align(baseN, INT4_ALIGN_SIZE);
    }
    const int32_t aTypeSize = DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType);
    const int32_t bTypeSize = DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType);
    if (baseM * baseK * aTypeSize / BITS_PER_BYTE > tilingIns_->bufferPool_.l0ASize / DB_ON) {
        singleCoreStatus.l0Status.dbL0A = DB_OFF;
    }
    if (baseN * baseK * bTypeSize / BITS_PER_BYTE> tilingIns_->bufferPool_.l0BSize / DB_ON) {
        singleCoreStatus.l0Status.dbL0B = DB_OFF;
    }
    if (baseM * baseN * FP32_BYTES > tilingIns_->bufferPool_.l0CSize / DB_ON) {
        singleCoreStatus.l0Status.dbL0C = DB_OFF;
    }
}

void MatmulTilingAlgorithm::GetMxUsedL1Size(const SingleCoreStatus& singleCoreStatus,
    int32_t& dataUsedL1Size, int32_t& scaleUsedL1Size, int32_t& biasUsedL1Size) const
{
    int32_t depthA1 = MathUtil::CeilDivision(singleCoreStatus.l1Status.kAL1, singleCoreStatus.l0Status.kL0) *
        singleCoreStatus.l1Status.mAL1 * singleCoreStatus.l1Status.dbAL1;
    int32_t depthB1 = MathUtil::CeilDivision(singleCoreStatus.l1Status.kBL1, singleCoreStatus.l0Status.kL0) *
        singleCoreStatus.l1Status.nBL1 * singleCoreStatus.l1Status.dbBL1;

    int32_t stepKa = MathUtil::CeilDivision(singleCoreStatus.l1Status.kAL1, singleCoreStatus.l0Status.kL0);
    int32_t stepKb = MathUtil::CeilDivision(singleCoreStatus.l1Status.kBL1, singleCoreStatus.l0Status.kL0);
    int32_t stepM = singleCoreStatus.l1Status.mAL1;
    int32_t stepN = singleCoreStatus.l1Status.nBL1;
    // A
    int32_t matrixByteSize = GetMatrixAByteSize() * DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
    int32_t stepSize = stepKa * stepM;
    int32_t cacheFactor = (depthA1 / stepSize - 1) % NUM_TWO;
    int32_t queDepth = cacheFactor == 0 ? DB_OFF : DB_ON;
    int32_t initBufferA1Size = queDepth * matrixByteSize * stepSize;
    // scaleA
    matrixByteSize = GetMatrixScaleAByteSize() * DTYPE_BIT_TAB.at(DataType::DT_FLOAT8_E8M0) / BITS_PER_BYTE;
    int32_t initBufferScaleA1Size = queDepth * matrixByteSize * stepSize;

    // B
    matrixByteSize = GetMatrixBByteSize() * DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
    stepSize = stepKb * stepN;
    cacheFactor = (depthB1 / stepSize - 1) % NUM_TWO;
    queDepth = cacheFactor == 0 ? DB_OFF : DB_ON;
    int32_t initBufferB1Size = queDepth * matrixByteSize * stepSize;
    // scaleB
    matrixByteSize = GetMatrixScaleBByteSize() * DTYPE_BIT_TAB.at(DataType::DT_FLOAT8_E8M0) / BITS_PER_BYTE;
    int32_t initBufferScaleB1Size = queDepth * matrixByteSize * stepSize;

    dataUsedL1Size = initBufferA1Size + initBufferB1Size;
    // scale is fp8e8m0, scaleFactorKa/scaleFactorKb/scaleFactorM/scaleFactorN is 1
    scaleUsedL1Size = initBufferScaleA1Size + initBufferScaleB1Size;
    // bias is fp32
    int32_t bias = tilingIns_->isBias ? 1 : 0;
    int32_t baseN = singleCoreStatus.l0Status.nL0 * C0_SIZE;
    biasUsedL1Size = bias * baseN * DTYPE_BIT_TAB.at(tilingIns_->biasType_.dataType) / BITS_PER_BYTE;
}

void MatmulTilingAlgorithm::AdjustSparseL0Factors(SingleCoreStatus& singleCoreStatus) const
{
    // determine whether the scenario is sparse
    if (!tilingIns_->isSparse_) {
        TILING_LOG_DEBUG("Not sparse scenario does not need to adjust L0Factors.");
        return;
    }

    int32_t baseK =
       singleCoreStatus.l0Status.kL0 * (C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);
    constexpr int32_t sparseBaseKFac = 64; // baseK need to align to 64 on Sparse
    if (baseK <= sparseBaseKFac) {
        baseK = sparseBaseKFac;
    } else {
        baseK = MathUtil::AlignDown(baseK, sparseBaseKFac);
    }
    singleCoreStatus.l0Status.kL0 =
        baseK / (C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);

    // check L0A/L0B/L0Csize for L0 DB
    CheckL0DB(singleCoreStatus, baseK);
}

void MatmulTilingAlgorithm::UpdateBaseKForMxGemv(int32_t &baseK, SingleCoreStatus& singleCoreStatus) const
{
    // For Gemv with m=1, baseK must align to 1024B to meet scaleA's 32B alignment from L1 to L0 cache.
    if (tilingIns_->aType_.type == CubeFormat::VECTOR || tilingIns_->aType_.scaleType == CubeFormat::VECTOR) {
        baseK = baseK <= MX_L1_TO_L0_ALIGN ? MX_L1_TO_L0_ALIGN : MathUtil::AlignDown(baseK, MX_L1_TO_L0_ALIGN);
        singleCoreStatus.l0Status.kL0 =
            baseK / (C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);
        int32_t baseN = tilingIns_->bufferPool_.l0BSize /
            (baseK * DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE * singleCoreStatus.l0Status.dbL0B);
        singleCoreStatus.l0Status.nL0 =
            singleCoreStatus.l0Status.nL0 >= baseN / C0_SIZE ? baseN / C0_SIZE : singleCoreStatus.l0Status.nL0;
    }
}

void MatmulTilingAlgorithm::AdjustMxL0Factors(SingleCoreStatus& singleCoreStatus) const
{
    // Determine whether the scenario is MX.
    if (tilingIns_->madType_ != MatrixMadType::MXMODE) {
        return;
    }
    if (!tilingIns_->aType_.hasSetScaleType) {
        tilingIns_->aType_.scalePos = tilingIns_->aType_.pos;
        tilingIns_->aType_.scaleType = tilingIns_->aType_.type;
        tilingIns_->aType_.isScaleTrans = tilingIns_->aType_.isTrans;
    }
    if (!tilingIns_->bType_.hasSetScaleType) {
        tilingIns_->bType_.scalePos = tilingIns_->bType_.pos;
        tilingIns_->bType_.scaleType = tilingIns_->bType_.type;
        tilingIns_->bType_.isScaleTrans = tilingIns_->bType_.isTrans;
    }
    // In the NZ scenario, ensure that the base size of the inner axis is 64-aligned downwards.
    constexpr int32_t l0Factor = INT4_ALIGN_SIZE / C0_SIZE;
    if (tilingIns_->aType_.type == CubeFormat::NZ && tilingIns_->aType_.isTrans) {
        if (singleCoreStatus.l0Status.mL0 > l0Factor) {
            singleCoreStatus.l0Status.mL0 = singleCoreStatus.l0Status.mL0 / l0Factor * l0Factor;
        } else {
            singleCoreStatus.l0Status.mL0 = MathUtil::Align(singleCoreStatus.l0Status.mL0, L0_FACTOR_NUM_LIMIT);
        }
    }
    if (tilingIns_->bType_.type == CubeFormat::NZ && !tilingIns_->bType_.isTrans) {
        if (singleCoreStatus.l0Status.nL0 > l0Factor) {
            singleCoreStatus.l0Status.nL0 = singleCoreStatus.l0Status.nL0 / l0Factor * l0Factor;
        } else {
            singleCoreStatus.l0Status.nL0 = MathUtil::Align(singleCoreStatus.l0Status.nL0, L0_FACTOR_NUM_LIMIT);
        }
    }
    // FP8 baseK need must be 64 element aligned
    int32_t baseK = singleCoreStatus.l0Status.kL0 * (C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);
    if ((tilingIns_->aType_.dataType == DataType::DT_FLOAT8_E5M2 || tilingIns_->aType_.dataType == DataType::DT_FLOAT8_E4M3FN) &&
        (tilingIns_->bType_.dataType == DataType::DT_FLOAT8_E5M2 || tilingIns_->bType_.dataType == DataType::DT_FLOAT8_E4M3FN)) {
        baseK = baseK <= MX_BASEK_FACTOR ? MX_BASEK_FACTOR : MathUtil::AlignDown(baseK, MX_BASEK_FACTOR);
        singleCoreStatus.l0Status.kL0 = 
            baseK / (C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);
    }
    UpdateBaseKForMxGemv(baseK, singleCoreStatus); // mx gemv baseK must align 1024
    bool mL0NeedAlign = tilingIns_->aType_.type == CubeFormat::ND && tilingIns_->aType_.isTrans &&
        tilingIns_->aType_.scalePos == TPosition::TSCM;
    if (mL0NeedAlign) {
        singleCoreStatus.l0Status.mL0 = MathUtil::Align(singleCoreStatus.l0Status.mL0, L0_FACTOR_NUM_LIMIT);
    }
    bool nL0NeedAlign = tilingIns_->bType_.type == CubeFormat::ND && !tilingIns_->bType_.isTrans &&
        tilingIns_->bType_.scalePos == TPosition::TSCM;
    if (nL0NeedAlign) {
        singleCoreStatus.l0Status.nL0 = MathUtil::Align(singleCoreStatus.l0Status.nL0, L0_FACTOR_NUM_LIMIT);
    }
    // check L0A/L0B/L0CSize for L0DB
    CheckL0DB(singleCoreStatus, baseK);
}

void MatmulTilingAlgorithm::AdjustMxL1Factors(SingleCoreStatus& singleCoreStatus) const
{
    // determine whether the scenario is MX
    if (tilingIns_->madType_ != MatrixMadType::MXMODE) {
        return;
    }
    int32_t dataUsedL1Size = 0;
    int32_t scaleUsedL1Size = 0;
    int32_t biasUsedL1Size = 0;
    GetMxUsedL1Size(singleCoreStatus, dataUsedL1Size, scaleUsedL1Size, biasUsedL1Size);
    // The existing tiling policy causes the L1 threshold to exceed the threshold.
    // Adjust the tiling policy to the basic one. That is, only baseM * baseK + baseN * baseK is cached ai L1.
    if (dataUsedL1Size + scaleUsedL1Size + biasUsedL1Size > tilingIns_->bufferPool_.l1Size) {
        // checks whether the tiling is valid.
        // If the tiling is invalid, the system uses the minimum tiling policy.
        singleCoreStatus.l1Status.kAL1 = singleCoreStatus.l0Status.kL0;
        singleCoreStatus.l1Status.kBL1 = singleCoreStatus.l0Status.kL0;
        singleCoreStatus.l1Status.mAL1 = 1;
        singleCoreStatus.l1Status.nBL1 = 1;
    }
}

void MatmulTilingAlgorithm::FixMxScaleFactorByRange(uint8_t& factor, uint8_t maxFactor) const
{
    factor = factor < maxFactor ? factor : maxFactor;
    // scaleFactor is in range of [1, 127]
    factor = factor > static_cast<uint8_t>(1) ? factor : static_cast<uint8_t>(1);
    factor = factor < SCALE_FACTOR_MAX_VALUE ? factor : SCALE_FACTOR_MAX_VALUE;
}

void MatmulTilingAlgorithm::FixMxScaleFactorByPosition(uint8_t& scaleFactorM, uint8_t& scaleFactorN, uint8_t& scaleFactorKa, uint8_t& scaleFactorKb) const
{
    if ((tilingIns_->aType_.type == CubeFormat::ND && tilingIns_->aType_.isTrans == true && 
        tilingIns_->aType_.scalePos == TPosition::TSCM) &&
        (tilingIns_->bType_.type == CubeFormat::ND && tilingIns_->bType_.isTrans == false && 
        tilingIns_->bType_.scalePos == TPosition::TSCM)) {
        scaleFactorM = static_cast<uint8_t>(1);
        scaleFactorN = static_cast<uint8_t>(1);
        scaleFactorKa = static_cast<uint8_t>(1);
        scaleFactorKb = static_cast<uint8_t>(1);
    }

    if (tilingIns_->aType_.scalePos == TPosition::TSCM) {
        scaleFactorM = static_cast<uint8_t>(1);
        scaleFactorKa = static_cast<uint8_t>(1);
    }

    if (tilingIns_->bType_.scalePos == TPosition::TSCM) {
        scaleFactorN = static_cast<uint8_t>(1);
        scaleFactorKb = static_cast<uint8_t>(1);
    }
}

void MatmulTilingAlgorithm::GetMxScaleSize(int32_t& scaleA1Size, int32_t& scaleB1Size) const
{
    if (tilingIns_->aType_.scalePos == TPosition::TSCM) {
        scaleA1Size = MathUtil::Align(tilingIns_->tiling_.get_singleCoreM(), BLOCK_CUBE) *
            (MathUtil::CeilDivision(tilingIns_->tiling_.get_singleCoreK(), MX_BASEK_FACTOR) * NUM_TWO);
    } else {
        scaleA1Size = tilingIns_->tiling_.get_stepKa() * tilingIns_->tiling_.get_stepM() * GetMatrixScaleAByteSize();
    }

    if (tilingIns_->bType_.scalePos == TPosition::TSCM) {
        scaleB1Size = MathUtil::Align(tilingIns_->tiling_.get_singleCoreN(), BLOCK_CUBE) *
            (MathUtil::CeilDivision(tilingIns_->tiling_.get_singleCoreK(), MX_BASEK_FACTOR) * NUM_TWO);
    } else {
        scaleB1Size = tilingIns_->tiling_.get_stepKb() * tilingIns_->tiling_.get_stepN() * GetMatrixScaleBByteSize();
    }
}


void MatmulTilingAlgorithm::GetMxScaleFactor(const SingleCoreStatus& singleCoreStatus, int32_t& mxTypePara) const
{
    int32_t dataUsedL1Size = 0;
    int32_t scaleUsedL1Size = 0;
    int32_t biasUsedL1Size = 0;
    GetMxUsedL1Size(singleCoreStatus, dataUsedL1Size, scaleUsedL1Size, biasUsedL1Size);

    int32_t scaleA1Size = 0;
    int32_t scaleB1Size = 0;
    GetMxScaleSize(scaleA1Size, scaleB1Size);

    int32_t remainedL1BufferSize = (tilingIns_->bufferPool_.l1Size - (dataUsedL1Size + biasUsedL1Size + scaleUsedL1Size)) / MX_L1_BUFFER_NUM;
    int32_t kStep = MathUtil::CeilDivision(tilingIns_->tiling_.get_singleCoreK(), tilingIns_->tiling_.get_baseK());

    uint8_t scaleFactorKa = static_cast<uint8_t>(remainedL1BufferSize / scaleA1Size + 1);
    uint8_t maxScaleFactorKa = static_cast<uint8_t>(MathUtil::CeilDivision(kStep, tilingIns_->tiling_.get_stepKa()));
    FixMxScaleFactorByRange(scaleFactorKa, maxScaleFactorKa);

    uint8_t scaleFactorKb = static_cast<uint8_t>(remainedL1BufferSize / scaleB1Size + 1);
    uint8_t maxScaleFactorKb = static_cast<uint8_t>(MathUtil::CeilDivision(kStep, tilingIns_->tiling_.get_stepKb()));
    FixMxScaleFactorByRange(scaleFactorKb, maxScaleFactorKb);

    uint8_t scaleFactorM = 1;
    if (scaleFactorKa == maxScaleFactorKa) {
        scaleFactorM = static_cast<uint8_t>(remainedL1BufferSize / (static_cast<int32_t>(scaleFactorKa) * scaleA1Size));
        int32_t mStep = MathUtil::CeilDivision(tilingIns_->tiling_.get_singleCoreM(), tilingIns_->tiling_.get_baseM());
        uint8_t maxScaleFactorM = static_cast<uint8_t>(MathUtil::CeilDivision(mStep, tilingIns_->tiling_.get_stepM()));
        FixMxScaleFactorByRange(scaleFactorM, maxScaleFactorM);
    }

    uint8_t scaleFactorN = 1;
    if (scaleFactorKb == maxScaleFactorKb) {
        scaleFactorN = static_cast<uint8_t>(remainedL1BufferSize / (static_cast<int32_t>(scaleFactorKb) * scaleB1Size));
        int32_t nStep = MathUtil::CeilDivision(tilingIns_->tiling_.get_singleCoreN(), tilingIns_->tiling_.get_baseN());
        uint8_t maxScaleFactorN = static_cast<uint8_t>(MathUtil::CeilDivision(nStep, tilingIns_->tiling_.get_stepN()));
        FixMxScaleFactorByRange(scaleFactorN, maxScaleFactorN);
    }

    FixMxScaleFactorByPosition(scaleFactorM, scaleFactorN, scaleFactorKa, scaleFactorKb);

    scaleA1Size = static_cast<int32_t>(scaleFactorKa) * scaleA1Size;
    scaleB1Size = static_cast<int32_t>(scaleFactorKb) * scaleB1Size;

    if (dataUsedL1Size + biasUsedL1Size + scaleA1Size + scaleB1Size > tilingIns_->bufferPool_.l1Size) {
        scaleFactorM = static_cast<uint8_t>(1);
        scaleFactorKa = static_cast<uint8_t>(1);
        scaleFactorN = static_cast<uint8_t>(1);
        scaleFactorKb = static_cast<uint8_t>(1);
    }
    // 8bit: 0~6bit:scaleFactor, 7bit(reserved):double buffer flag
    scaleFactorKa = scaleFactorKa & static_cast<uint8_t>(0x7F);
    scaleFactorKb = scaleFactorKb & static_cast<uint8_t>(0x7F);
    scaleFactorM = scaleFactorM & static_cast<uint8_t>(0x7F);
    scaleFactorN = scaleFactorN & static_cast<uint8_t>(0x7F);
    mxTypePara = static_cast<int32_t>(static_cast<uint32_t>(mxTypePara) | scaleFactorKa);
    mxTypePara = static_cast<int32_t>(static_cast<uint32_t>(mxTypePara) | static_cast<uint32_t>(scaleFactorKb << 8U));
    mxTypePara = static_cast<int32_t>(static_cast<uint32_t>(mxTypePara) | static_cast<uint32_t>(scaleFactorM << 16U));
    mxTypePara = static_cast<int32_t>(static_cast<uint32_t>(mxTypePara) | static_cast<uint32_t>(scaleFactorN << 24U));
}

void MatmulTilingAlgorithm::PreprocessL0DB()
{
    dbL0A_ = g_tempCfg.l0aDB;
    dbL0B_ = g_tempCfg.l0bDB;
    dbL0C_ = g_tempCfg.l0cDB;
    if (tilingIns_->baseM != -1) {
        const int32_t baseLeftSize = tilingIns_->baseM * C0_BYTE_SIZE;
        if (baseLeftSize > tilingIns_->bufferPool_.l0ASize / DB_ON) {
            dbL0A_ = DB_OFF;
        }
    }
    if (tilingIns_->baseN != -1) {
        const int32_t baseRightSize = tilingIns_->baseN * C0_BYTE_SIZE;
        if (baseRightSize > tilingIns_->bufferPool_.l0BSize / DB_ON) {
            dbL0B_ = DB_OFF;
        }
    }
    if (tilingIns_->baseM != -1 && tilingIns_->baseN != -1) {
        const int32_t baseMatrixSize = tilingIns_->baseM * tilingIns_->baseN * C0_BYTE_SIZE;
        if (baseMatrixSize > tilingIns_->bufferPool_.l0CSize / DB_ON) {
            dbL0C_ = DB_OFF;
        }
    }
    return;
}

void MatmulTilingAlgorithm::SetDepthL1CacheUBParams(int32_t &a1LengthCache, int32_t &b1LengthCache) const
{
    if (!tilingIns_->enableL1CacheUB ||
        tilingIns_->socVersion != platform_ascendc::SocVersion::ASCEND310P) {
        return;
    }
    int32_t a1Length = tilingIns_->tiling_.get_baseM() * tilingIns_->tiling_.get_baseK() *
                       DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
    int32_t b1Length = tilingIns_->tiling_.get_baseN() * tilingIns_->tiling_.get_baseK() *
                       DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
    a1LengthCache = a1Length * tilingIns_->tiling_.get_stepKa() * tilingIns_->tiling_.get_stepM();
    b1LengthCache = b1Length * tilingIns_->tiling_.get_stepKb() * tilingIns_->tiling_.get_stepN();
    int32_t freeL1Size = tilingIns_->bufferPool_.l1Size - tilingIns_->tiling_.get_depthA1() * a1Length -
                         tilingIns_->tiling_.get_depthB1() * b1Length;
    if (freeL1Size <= 0) {
        return;
    }
    const int32_t splitNum = 2;
    int32_t aOrgShapeSize = tilingIns_->tiling_.get_singleCoreM() * tilingIns_->tiling_.get_singleCoreK();
    int32_t bOrgShapeSize = tilingIns_->tiling_.get_singleCoreN() * tilingIns_->tiling_.get_singleCoreK();

    if ((tilingIns_->aType_.type == CubeFormat::ND && tilingIns_->aType_.pos != TPosition::TSCM) &&
        (tilingIns_->bType_.type == CubeFormat::ND && tilingIns_->bType_.pos != TPosition::TSCM)) {
        bool aFullLoad = false;
        bool bFullLoad = false;
        aFullLoad = aOrgShapeSize > 0 && aOrgShapeSize < freeL1Size / splitNum;
        bFullLoad = bOrgShapeSize > 0 && bOrgShapeSize < freeL1Size / splitNum;
        if (aFullLoad && bFullLoad) {
            tilingIns_->tiling_.set_depthAL1CacheUB(1);
            tilingIns_->tiling_.set_depthBL1CacheUB(1);
            a1LengthCache = aOrgShapeSize; // update
            b1LengthCache = bOrgShapeSize;
        } else if (aFullLoad) {
            tilingIns_->tiling_.set_depthAL1CacheUB(1);
            a1LengthCache = aOrgShapeSize;
            int32_t depthL1CacheUB = b1LengthCache > 0 ? (freeL1Size - aOrgShapeSize) / b1LengthCache : 0;
            tilingIns_->tiling_.set_depthBL1CacheUB(depthL1CacheUB);
        } else if (bFullLoad) {
            tilingIns_->tiling_.set_depthBL1CacheUB(1);
            b1LengthCache = bOrgShapeSize;
            int32_t depthL1CacheUB = a1LengthCache > 0 ? (freeL1Size - bOrgShapeSize) / a1LengthCache : 0;
            tilingIns_->tiling_.set_depthAL1CacheUB(depthL1CacheUB);
        } else {
            if (a1LengthCache > freeL1Size) {
                int32_t depthBL1CacheUB = b1LengthCache > 0 ? freeL1Size / b1LengthCache : 0;
                tilingIns_->tiling_.set_depthBL1CacheUB(depthBL1CacheUB);
            } else if (b1LengthCache > freeL1Size) {
                int32_t depthAL1CacheUB = a1LengthCache > 0 ? freeL1Size / a1LengthCache : 0;
                tilingIns_->tiling_.set_depthAL1CacheUB(depthAL1CacheUB);
            } else if (a1LengthCache <= freeL1Size / splitNum && b1LengthCache <= freeL1Size / splitNum) {
                int32_t depthAL1CacheUB = a1LengthCache > 0 ? freeL1Size / splitNum / a1LengthCache : 0;
                int32_t depthBL1CacheUB = b1LengthCache > 0 ? freeL1Size / splitNum / b1LengthCache : 0;
                tilingIns_->tiling_.set_depthAL1CacheUB(depthAL1CacheUB);
                tilingIns_->tiling_.set_depthBL1CacheUB(depthBL1CacheUB);
            } else {
                // can only cache one matrix
                if (a1LengthCache <= b1LengthCache) {
                    tilingIns_->tiling_.set_depthAL1CacheUB(freeL1Size / a1LengthCache);
                } else {
                    tilingIns_->tiling_.set_depthBL1CacheUB(freeL1Size / b1LengthCache);
                }
            }
        }
    } else if (tilingIns_->aType_.type == CubeFormat::ND && tilingIns_->aType_.pos != TPosition::TSCM) {
        if (aOrgShapeSize > 0 && aOrgShapeSize < freeL1Size) {
            tilingIns_->tiling_.set_depthAL1CacheUB(1);
            a1LengthCache = aOrgShapeSize;
        } else if (a1LengthCache > 0) {
            tilingIns_->tiling_.set_depthAL1CacheUB(freeL1Size / a1LengthCache);
        }
    } else if (tilingIns_->bType_.type == CubeFormat::ND && tilingIns_->bType_.pos != TPosition::TSCM) {
        if (bOrgShapeSize > 0 && bOrgShapeSize < freeL1Size) {
            tilingIns_->tiling_.set_depthBL1CacheUB(1);
            b1LengthCache = bOrgShapeSize;
        } else if (b1LengthCache > 0) {
            tilingIns_->tiling_.set_depthBL1CacheUB(freeL1Size / b1LengthCache);
        }
    } else {
        return;
    }
}

int MatmulTilingAlgorithm::UpdateDepthB1(const SingleCoreStatus& singleCoreStatus) const
{
    int depthB1 = MathUtil::CeilDivision(singleCoreStatus.l1Status.kBL1, singleCoreStatus.l0Status.kL0) *
        singleCoreStatus.l1Status.nBL1 * singleCoreStatus.l1Status.dbBL1;
    // only bType is f32 need update
    if (tilingIns_->bType_.dataType != DataType::DT_FLOAT
        || tilingIns_->socVersion != platform_ascendc::SocVersion::ASCEND910B) {
        return depthB1;
    }
    uint16_t alignedBaseK = MathUtil::CeilDivision(tilingIns_->baseK, FP32_ALIGN_SIZE) * FP32_ALIGN_SIZE;
    uint16_t alignedBaseKN = alignedBaseK * tilingIns_->baseN;

    uint16_t alignedBaseKM = tilingIns_->baseK * tilingIns_->baseM;
    if (tilingIns_->aType_.isTrans && tilingIns_->aType_.dataType == DataType::DT_FLOAT) {
        alignedBaseKM = alignedBaseK * tilingIns_->baseM;
    }
    // if L1 size is overflow, decrease depthB1
    if ((tilingIns_->tiling_.get_depthA1() *alignedBaseKM + alignedBaseKN * depthB1) * sizeof(float)
        > static_cast<size_t>(tilingIns_->bufferPool_.l1Size)) {
        depthB1 = tilingIns_->baseN * tilingIns_->baseK * depthB1 / alignedBaseKN;
        depthB1 = depthB1 < 1 ? 1 : depthB1;
    }
    return depthB1;
}
int32_t MatmulTilingAlgorithm::GetSingleM() const
{
    return tilingIns_->singleM != -1 ? tilingIns_->singleM : tilingIns_->orgM;
}
int32_t MatmulTilingAlgorithm::GetSingleN() const
{
    return tilingIns_->singleN != -1 ? tilingIns_->singleN : tilingIns_->orgN;
}
int32_t MatmulTilingAlgorithm::GetSingleK() const
{
    return tilingIns_->singleK != -1 ? tilingIns_->singleK : tilingIns_->orgKa;
}
void MatmulTilingAlgorithm::GetSingleShape(const CoreStatusPack &coreStatus, const MatmulRunParas &param,
    int32_t &singleCoreM, int32_t &singleCoreN, int32_t &singleCoreK) const
{
    singleCoreM = GetSingleM();
    singleCoreM = MathUtil::CeilDivision(singleCoreM, coreStatus.mDim);
    singleCoreN = GetSingleN();
    singleCoreN = MathUtil::CeilDivision(singleCoreN, coreStatus.nDim);
    singleCoreK = GetSingleK();
    singleCoreK = MathUtil::CeilDivision(singleCoreK, coreStatus.kDim);
    if (enableSingleShape_) {
        singleCoreM = tilingIns_->singleCoreM != -1 ? tilingIns_->singleCoreM : tilingIns_->singleM;
        singleCoreN = tilingIns_->singleCoreN != -1 ? tilingIns_->singleCoreN : tilingIns_->singleN;
        singleCoreK = tilingIns_->singleCoreK != -1 ? tilingIns_->singleCoreK : tilingIns_->singleK;
    }
    if (numOfBlock_ > 1) {
        int32_t aAlignSize = DATA_COPY_ALIGN_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE;
        int32_t bAlignSize = DATA_COPY_ALIGN_SIZE / DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) * BITS_PER_BYTE;
        auto multiCoreScenario = GetMultiCoreScenario(param);
        bool needAlign = multiCoreScenario == MultiCoreScenario::SPLIT_MN ||
                        multiCoreScenario == MultiCoreScenario::SPLIT_SMALL_MN ||
                        tilingIns_->aType_.type == CubeFormat::NZ ||
                        tilingIns_->bType_.type == CubeFormat::NZ;
        bool needOutputAlign = NeedOutputAlign(singleCoreM, singleCoreN, singleCoreK);
        bool singleMNAlign = (tilingIns_->cType_.type == CubeFormat::NZ);
        (void)AlignSingleShape((needAlign && (!tilingIns_->bType_.isTrans || needOutputAlign)) || singleMNAlign,
            param.n32 * C0_SIZE, coreStatus.nDim, bAlignSize, singleCoreN);
        (void)AlignSingleShape((needAlign && tilingIns_->aType_.isTrans) || singleMNAlign,
            param.m32 * C0_SIZE, coreStatus.mDim, aAlignSize, singleCoreM);
        if (tilingIns_->enableSplitK_) {
            if (tilingIns_->aType_.dataType == DataType::DT_FLOAT ||
                tilingIns_->bType_.dataType == DataType::DT_FLOAT) {
                singleCoreK = MathUtil::CeilDivision(param.k32, coreStatus.kDim) * FLOAT32_REDUCE_BLOCK_SIZE;
            } else if ((tilingIns_->aType_.dataType == DataType::DT_INT8 ||
                tilingIns_->bType_.dataType == DataType::DT_INT8)) {
                singleCoreK = MathUtil::CeilDivision(param.k32, coreStatus.kDim) * INT8_REDUCE_BLOCK_SIZE;
            } else if ((tilingIns_->aType_.dataType == DataType::DT_INT4 ||
                tilingIns_->bType_.dataType == DataType::DT_INT4)) {
                singleCoreK = MathUtil::CeilDivision(param.k32, coreStatus.kDim) * INT4_REDUCE_BLOCK_SIZE;
            } else {
                singleCoreK = MathUtil::CeilDivision(param.k32, coreStatus.kDim) * REDUCE_BLOCK_SIZE;
            }
        }
    }
}

bool MatmulTilingAlgorithm::CheckSingleShape(int32_t singleCoreM, int32_t singleCoreN, int32_t singleCoreK) const
{
    (void)singleCoreM;
    (void)singleCoreK;
    if (tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND910 ||
        tilingIns_->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        // ub only can process with 32B aligned, if format is ND, and D non-aligned output can't pad
        if (tilingIns_->cType_.pos == TPosition::VECCALC && tilingIns_->cType_.type == CubeFormat::ND &&
            (singleCoreN * DTYPE_BYTE_TAB.at(tilingIns_->cType_.dataType)) % C0_BYTE_SIZE != 0) {
            TILING_LOG_INFO("For ascend310p/ascend910, when matrix c pos is VECCACL and singleCoreN is not 32B "
                             "aligned, matrix c not support ND format.");
            return false;
        }
    }
    return true;
}

bool MatmulTilingAlgorithm::CheckFixSplitInputs(int32_t singleCoreM) const
{
    if (tilingIns_->baseM == -1) {
        return true;
    }

    if (tilingIns_->baseM == 0) {
        TILING_LOG_WARNING("MatmulApi Tiling : baseM should be larger than zero.");
        return false;
    }
    int32_t blockM = MathUtil::CeilDivision(singleCoreM, tilingIns_->baseM);
    if (blockM > N_BUFFER_33_FACTOR) {
        TILING_LOG_WARNING("MatmulApi Tiling : Ceil(singleCoreM / baseM) = %d, should be less than or equal to 3.", blockM);
        return false;
    }
    return true;
}

void MatmulTilingAlgorithm::CalcBaseShape(const SingleCoreStatus &singleCoreStatus,
    int32_t &baseM, int32_t &baseN, int32_t &baseK, int32_t &reduceSize) const
{
    baseM = singleCoreStatus.l0Status.mL0 * C0_SIZE;
    baseM = tilingIns_->baseM != -1 ? tilingIns_->baseM : baseM;
    if (tilingIns_->aType_.type == CubeFormat::ND && tilingIns_->aType_.isTrans &&
        tilingIns_->aType_.scalePos == TPosition::TSCM) {
        baseM = MathUtil::Align(singleCoreStatus.l0Status.mL0, L0_FACTOR_NUM_LIMIT) * C0_SIZE;
    }

    baseN = singleCoreStatus.l0Status.nL0 * C0_SIZE;
    baseN = tilingIns_->baseN != -1 ? tilingIns_->baseN : baseN;
    if (tilingIns_->bType_.type == CubeFormat::ND && !tilingIns_->bType_.isTrans &&
        tilingIns_->bType_.scalePos == TPosition::TSCM) {
        baseN = MathUtil::Align(singleCoreStatus.l0Status.nL0, L0_FACTOR_NUM_LIMIT) * C0_SIZE;
    }

    reduceSize = static_cast<int32_t>(C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);
    baseK = singleCoreStatus.l0Status.kL0 * reduceSize;
}

bool MatmulTilingAlgorithm::CheckL0ASize(int32_t singleCoreM, int32_t singleCoreK, int32_t &baseM, int32_t &baseK) const
{
    int32_t l0aLoadSize = baseM * baseK * DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
    if (l0aLoadSize <= tilingIns_->bufferPool_.l0ASize / DB_OFF) {
        return true;
    }

    if (tilingIns_->baseM == -1) {
        int32_t blockM = MathUtil::CeilDivision(singleCoreM, baseM);
        while (blockM < N_BUFFER_33_FACTOR) {
            blockM++;
            baseM = MathUtil::Align(MathUtil::CeilDivision(singleCoreM, blockM), C0_SIZE);
            l0aLoadSize = baseM * baseK * DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
            if (l0aLoadSize <= tilingIns_->bufferPool_.l0ASize / DB_OFF) {
                return true;
            }
        }
    }

    int32_t blockK = MathUtil::CeilDivision(singleCoreK, baseK);
    int32_t reduceSize =
        static_cast<int32_t>(C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);
    while (blockK < N_BUFFER_33_FACTOR) {
        blockK++;
        baseK = MathUtil::Align(MathUtil::CeilDivision(singleCoreK, blockK), reduceSize);
        l0aLoadSize = baseM * baseK * DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
        if (l0aLoadSize <= tilingIns_->bufferPool_.l0ASize / DB_OFF) {
            return true;
        }
    }

    TILING_LOG_WARNING("MatmulApi Tiling : L0A load size %d with baseM %d and baseK %d as final attempt exceeds L0ASize %d. "
        "Cannot find valid baseM & baseK under current singleCoreM %d and singleCoreK %d.",
        l0aLoadSize, baseM, baseK, tilingIns_->bufferPool_.l0ASize, singleCoreM, singleCoreK);
    return false;
}

bool MatmulTilingAlgorithm::CheckL0BSize(int32_t singleCoreN, int32_t singleCoreK, int32_t &baseN, int32_t &baseK) const
{
    int32_t l0bLoadSize = baseN * baseK * DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
    if (l0bLoadSize <= tilingIns_->bufferPool_.l0BSize / DB_OFF) {
        return true;
    }

    int32_t blockK = MathUtil::CeilDivision(singleCoreK, baseK);
    int32_t reduceSize =
        static_cast<int32_t>(C0_BYTE_SIZE / DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) * BITS_PER_BYTE);
    while (blockK < N_BUFFER_33_FACTOR) {
        blockK++;
        baseK = MathUtil::Align(MathUtil::CeilDivision(singleCoreK, blockK), reduceSize);
        l0bLoadSize = baseN * baseK * DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
        if (l0bLoadSize <= tilingIns_->bufferPool_.l0BSize / DB_OFF) {
            return true;
        }
    }

    if (tilingIns_->baseN == -1) {
        auto kSize = baseK * static_cast<int32_t>(DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE);
        if (kSize > 0) {
            baseN = C0_SIZE * (tilingIns_->bufferPool_.l0BSize / kSize / C0_SIZE);
            l0bLoadSize = baseN * baseK * DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
            if (l0bLoadSize <= tilingIns_->bufferPool_.l0BSize / DB_OFF) {
                return true;
            }
        }
    }

    TILING_LOG_WARNING("MatmulApi Tiling : L0B load size %d with baseN %d and baseK %d as final attempt exceeds L0BSize %d. "
        "Cannot find valid baseN & baseK under current singleCoreN %d and singleCoreK %d.",
        l0bLoadSize, baseN, baseK, tilingIns_->bufferPool_.l0BSize, singleCoreN, singleCoreK);
    return false;
}

bool MatmulTilingAlgorithm::CheckL0CSize(int32_t singleCoreM, int32_t singleCoreN, int32_t &baseM, int32_t &baseN) const
{
    int32_t l0cLoadSize = baseM * baseN * FP32_BYTES;
    if (l0cLoadSize <= tilingIns_->bufferPool_.l0CSize / DB_OFF) {
        return true;
    }

    if (tilingIns_->baseM == -1) {
        int32_t blockM = MathUtil::CeilDivision(singleCoreM, baseM);
        while (blockM < N_BUFFER_33_FACTOR) {
            blockM++;
            baseM = MathUtil::Align(MathUtil::CeilDivision(singleCoreM, blockM), C0_SIZE);
            l0cLoadSize = baseM * baseN * FP32_BYTES;
            if (l0cLoadSize <= tilingIns_->bufferPool_.l0CSize / DB_OFF) {
                return true;
            }
        }
    }

    if (tilingIns_->baseN == -1) {
        baseN = min(baseN, C0_SIZE * (tilingIns_->bufferPool_.l0CSize / (baseM * FP32_BYTES) / C0_SIZE));
        l0cLoadSize = baseM * baseN * FP32_BYTES;
        if (l0cLoadSize <= tilingIns_->bufferPool_.l0CSize / DB_OFF) {
            return true;
        }
    }

    TILING_LOG_WARNING("MatmulApi Tiling : L0C load size %d with baseM %d and baseN %d as final attempt exceeds L0CSize %d. "
        "Cannot find valid baseM & baseN under current singleCoreM %d and singleCoreN %d.",
        l0cLoadSize, baseM, baseN, tilingIns_->bufferPool_.l0CSize, singleCoreM, singleCoreN);
    return false;
}

void MatmulTilingAlgorithm::CheckL0DB(int32_t baseM, int32_t baseN, int32_t baseK,
    SingleCoreStatus &singleCoreStatus) const
{
    int32_t l0aLoadSize = baseM * baseK * DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
    if (l0aLoadSize <= tilingIns_->bufferPool_.l0ASize / DB_ON) {
        singleCoreStatus.l0Status.dbL0A = DB_ON;
    } else {
        singleCoreStatus.l0Status.dbL0A = DB_OFF;
    }
    int32_t l0bLoadSize = baseN * baseK * DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
    if (l0bLoadSize <= tilingIns_->bufferPool_.l0BSize / DB_ON) {
        singleCoreStatus.l0Status.dbL0B = DB_ON;
    } else {
        singleCoreStatus.l0Status.dbL0B = DB_OFF;
    }
    int32_t l0cLoadSize = baseM * baseN * FP32_BYTES;
    if (l0cLoadSize <= tilingIns_->bufferPool_.l0CSize / DB_ON) {
        singleCoreStatus.l0Status.dbL0C = DB_ON;
    } else {
        singleCoreStatus.l0Status.dbL0C = DB_OFF;
    }
}

bool MatmulTilingAlgorithm::AdjustNBuffer33L0Factors(const MatmulRunParas &param, const CoreStatusPack &coreStatus,
    SingleCoreStatus &singleCoreStatus) const
{
    if (tilingIns_->scheduleType != ScheduleType::N_BUFFER_33) {
        TILING_LOG_DEBUG("No need to adjust L0Factors for non-nbuffer33 scenario.");
        return true;
    }
    int32_t singleCoreM = 0;
    int32_t singleCoreN = 0;
    int32_t singleCoreK = 0;
    GetSingleShape(coreStatus, param, singleCoreM, singleCoreN, singleCoreK);
    if (!CheckFixSplitInputs(singleCoreM)) {
        return false;
    }

    int32_t baseM = 0;
    int32_t baseN = 0;
    int32_t baseK = 0;
    int32_t reduceSize = 0;
    CalcBaseShape(singleCoreStatus, baseM, baseN, baseK, reduceSize);

    if (MathUtil::CeilDivision(singleCoreM, baseM) > N_BUFFER_33_FACTOR) {
        baseM = MathUtil::Align(MathUtil::CeilDivision(singleCoreM, N_BUFFER_33_FACTOR), C0_SIZE);
    }
    if (MathUtil::CeilDivision(singleCoreK, baseK) > N_BUFFER_33_FACTOR) {
        baseK = MathUtil::Align(MathUtil::CeilDivision(singleCoreK, N_BUFFER_33_FACTOR), reduceSize);
    }

    if (!CheckL0ASize(singleCoreM, singleCoreK, baseM, baseK)) {
        return false;
    }
    if (!CheckL0BSize(singleCoreN, singleCoreK, baseN, baseK)) {
        return false;
    }
    if (!CheckL0CSize(singleCoreM, singleCoreN, baseM, baseN)) {
        return false;
    }
    CheckL0DB(baseM, baseN, baseK, singleCoreStatus);

    if (tilingIns_->baseM == -1) {
        singleCoreStatus.l0Status.mL0 = MathUtil::CeilDivision(baseM, C0_SIZE);
    }
    if (tilingIns_->baseN == -1) {
        singleCoreStatus.l0Status.nL0 = MathUtil::CeilDivision(baseN, C0_SIZE);
    }
    singleCoreStatus.l0Status.kL0 = MathUtil::CeilDivision(baseK, reduceSize);
    return true;
}

int32_t MatmulTilingAlgorithm::GetNBuffer33L1Size(const SingleCoreStatus &singleCoreStatus) const
{
    int32_t curAL1Size = 0;
    int32_t depthA1 = MathUtil::CeilDivision(singleCoreStatus.l1Status.kAL1, singleCoreStatus.l0Status.kL0) *
        singleCoreStatus.l1Status.mAL1 * singleCoreStatus.l1Status.dbAL1;
    if (!MathUtil::CheckMulOverflow(tilingIns_->tiling_.get_baseM(), tilingIns_->tiling_.get_baseK(), curAL1Size) ||
        !MathUtil::CheckMulOverflow(curAL1Size, depthA1, curAL1Size) ||
        !MathUtil::CheckMulOverflow(curAL1Size, DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE,
        curAL1Size)) {
        return 0;
    }

    int32_t curBL1Size = 0;
    int32_t depthB1 = MathUtil::CeilDivision(singleCoreStatus.l1Status.kBL1, singleCoreStatus.l0Status.kL0) *
        singleCoreStatus.l1Status.nBL1 * singleCoreStatus.l1Status.dbBL1;
    if (!MathUtil::CheckMulOverflow(tilingIns_->tiling_.get_baseN(), tilingIns_->tiling_.get_baseK(), curBL1Size) ||
        !MathUtil::CheckMulOverflow(curBL1Size, depthB1, curBL1Size) ||
        !MathUtil::CheckMulOverflow(curBL1Size, DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE,
        curBL1Size)) {
        return 0;
    }

    int32_t biasL1Size = !tilingIns_->isBias ? 0 :
        tilingIns_->tiling_.get_baseN() * DTYPE_BIT_TAB.at(tilingIns_->biasType_.dataType) / BITS_PER_BYTE;
    int32_t dequantSize = 0;
    if (tilingIns_->deqType == DequantType::TENSOR) {
        dequantSize = singleCoreStatus.l1Status.nBL1 * tilingIns_->tiling_.get_baseN() * UINT64_TYPES;
    }
    return (curAL1Size + curBL1Size + biasL1Size + dequantSize);
}

bool MatmulTilingAlgorithm::AdjustNBuffer33L1Factors(const CoreStatusPack &coreStatus,
    SingleCoreStatus &singleCoreStatus) const
{
    if (tilingIns_->scheduleType != ScheduleType::N_BUFFER_33) {
        TILING_LOG_DEBUG("No need to adjust L1Factors for non-nbuffer33 scanario.");
        return true;
    }

    tilingIns_->tiling_.set_usedCoreNum(min(tilingIns_->tiling_.get_usedCoreNum(), tilingIns_->blockDim));
    if (coreStatus.batchDim * coreStatus.mDim * coreStatus.kDim * coreStatus.nDim > tilingIns_->blockDim) {
        tilingIns_->tiling_.set_singleCoreM(min(tilingIns_->tiling_.get_baseM() * N_BUFFER_33_FACTOR, GetSingleM()));
        tilingIns_->tiling_.set_singleCoreK(min(tilingIns_->tiling_.get_baseK() * N_BUFFER_33_FACTOR, GetSingleK()));
    }

    singleCoreStatus.l1Status.mAL1 =
        MathUtil::CeilDivision(tilingIns_->tiling_.get_singleCoreM(), tilingIns_->tiling_.get_baseM()); 
    int32_t blockK = MathUtil::CeilDivision(tilingIns_->tiling_.get_singleCoreK(), tilingIns_->tiling_.get_baseK());
    if (singleCoreStatus.l1Status.kAL1 != singleCoreStatus.l1Status.kBL1 ||
        MathUtil::CeilDivision(singleCoreStatus.l1Status.kAL1, singleCoreStatus.l0Status.kL0) != blockK) {
        singleCoreStatus.l1Status.kAL1 = blockK * singleCoreStatus.l0Status.kL0;
        singleCoreStatus.l1Status.kBL1 = singleCoreStatus.l1Status.kAL1;
    }

    singleCoreStatus.l1Status.dbAL1 = DB_OFF;
    singleCoreStatus.l1Status.dbBL1 = DB_OFF;
    int32_t curL1Size = GetNBuffer33L1Size(singleCoreStatus);
    if (curL1Size > tilingIns_->bufferPool_.l1Size) {
        bool succFlag = false;
        while (singleCoreStatus.l1Status.nBL1 > 1) {
            singleCoreStatus.l1Status.nBL1--;
            curL1Size = GetNBuffer33L1Size(singleCoreStatus);
            if (curL1Size <= tilingIns_->bufferPool_.l1Size) {
                succFlag = true;
                break;
            }
        }
        if (!succFlag) {
            TILING_LOG_WARNING("MatmulApi Tiling : Current L1 size %d exceeds L1Size limit %d. Cannot find a valid L1 tiling factors.",
                curL1Size, tilingIns_->bufferPool_.l1Size);
            return false;
        }
    }

    if (singleCoreStatus.l1Status.nBL1 <= 1) {
        return true;
    }
    int32_t nBL1DbOff = singleCoreStatus.l1Status.nBL1;
    singleCoreStatus.l1Status.dbBL1 = DB_ON;
    singleCoreStatus.l1Status.nBL1 = singleCoreStatus.l1Status.nBL1 / DB_ON;
    if (GetNBuffer33L1Size(singleCoreStatus) > tilingIns_->bufferPool_.l1Size) {
        singleCoreStatus.l1Status.dbBL1 = DB_OFF;
        singleCoreStatus.l1Status.nBL1 = nBL1DbOff;
    }
    return true;
}

int64_t MatmulTilingAlgorithm::Process()
{
    PreprocessL0DB();
    if (!CheckBaseMN()) {
        TILING_LOG_WARNING("MatmulApi Tiling : check baseM/baseN not success.");
        return -1;
    }
    enableSingleShape_ = false;
    splitCoreFlag_ = false;
    CoreStatusPack coreStatus;
    SingleCoreStatus singleCoreStatus;
    MatmulRunParas param;
    DimCalculator dimCalRes;
    FillParam(param);

    std::string opType = "MatMul";
    if (numOfBlock_ != 1) {
        NonFactorMap(opType, param, dimCalRes);
        if (DoMultiCoreSplitMNTiling(param, coreStatus, dimCalRes)) {
            return 0;
        }
        if (tilingIns_->scheduleType == ScheduleType::N_BUFFER_33) {
            return -1;
        }
        GetDims(opType, param, coreStatus, dimCalRes);
    } else {
        if (!g_tempCfg.factorSplit) {
            coreStatus.m = param.m32;
            coreStatus.k = param.k32;
            coreStatus.n = param.n32;
        } else {
            coreStatus.m = MathUtil::FindBestSingleCore(param.m32, param.mMapped, 1, false);
            coreStatus.k = MathUtil::FindBestSingleCore(param.k32, param.kMapped, 1, false);
            coreStatus.n = MathUtil::FindBestSingleCore(param.n32, param.nMapped, 1, false);
        }
        coreStatus.batchDim = 1;
        coreStatus.mDim = 1;
        coreStatus.kDim = 1;
        coreStatus.nDim = 1;
    }

    if (numOfBlock_ != 1 && tilingIns_->bType_.pos == TPosition::TSCM) {
        if (!splitCoreFlag_) {
            TILING_LOG_WARNING("MatmulApi Tiling : Multi core split B TSCM full loaded is not success.");
            return 1;
        }
    }
    // single-core logic
    GetL0Factors(opType, param, coreStatus, singleCoreStatus);
    AdjustSparseL0Factors(singleCoreStatus);
    AdjustMxL0Factors(singleCoreStatus);
    if (!AdjustNBuffer33L0Factors(param, coreStatus, singleCoreStatus)) {
        return -1;
    }
    if (singleCoreStatus.l0Status.mL0 == 0 || singleCoreStatus.l0Status.nL0 == 0 ||
        singleCoreStatus.l0Status.kL0 == 0) {
        TILING_LOG_WARNING("MatmulApi Tiling : ml0/nl0/kl0 is zero.");
        return -1;
    }
    GetL1Factors(opType, param, coreStatus, singleCoreStatus.l0Status, singleCoreStatus.l1Status);
    if (UpdateTiling(param, coreStatus, singleCoreStatus) == -1L) {
        return -1L;
    }
    const bool ans = CheckFinalParams(coreStatus);
    return ans ? 0 : -1;
}

int32_t MatmulTilingAlgorithm::GetC0Size(DataType dataType) const
{
    if (dataType == DataType::DT_FLOAT) {
        return FLOAT32_REDUCE_BLOCK_SIZE;
    } else if (DTYPE_BIT_TAB.at(dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8)) {
        return INT8_REDUCE_BLOCK_SIZE;
    } else if (DTYPE_BIT_TAB.at(dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT4)) {
        return INT4_REDUCE_BLOCK_SIZE;
    }
    return REDUCE_BLOCK_SIZE;
}

int32_t MatmulTilingAlgorithm::GetABaseHeightAlign(int32_t baseHeight) const
{
    if (tilingIns_->aType_.dataType == DataType::DT_FLOAT) {
        return MathUtil::Align(baseHeight, BLOCK_CUBE);
    } else if ((DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8) ||
        DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT4)) && tilingIns_->aType_.isTrans) {
        return MathUtil::Align(baseHeight, GetC0Size(tilingIns_->aType_.dataType));
    } else {
        return baseHeight;
    }
}

int32_t MatmulTilingAlgorithm::GetABaseWidthAlign(int32_t baseWidth) const
{
    if (tilingIns_->aType_.dataType == DataType::DT_FLOAT && tilingIns_->aType_.isTrans) {
        return MathUtil::Align(baseWidth, BLOCK_CUBE);
    } else if (tilingIns_->aType_.dataType == DataType::DT_FLOAT || (DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8) ||
        DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT4))) {
        return MathUtil::Align(baseWidth, GetC0Size(tilingIns_->aType_.dataType));
    } else {
        return baseWidth;
    }
}

int32_t MatmulTilingAlgorithm::GetBBaseHeightAlign(int32_t baseHeight) const
{
    if (tilingIns_->bType_.dataType == DataType::DT_FLOAT && !tilingIns_->bType_.isTrans) {
        return MathUtil::Align(baseHeight, BLOCK_CUBE);
    } else if (DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8) ||
        DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT4)) {
        return MathUtil::Align(baseHeight, GetC0Size(tilingIns_->bType_.dataType));
    } else {
        return baseHeight;
    }
}

int32_t MatmulTilingAlgorithm::GetBBaseWidthAlign(int32_t baseWidth) const
{
    if (tilingIns_->bType_.dataType == DataType::DT_FLOAT || ((DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT8) ||
        DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) == DTYPE_BIT_TAB.at(DataType::DT_INT4)) && !tilingIns_->bType_.isTrans)) {
        return MathUtil::Align(baseWidth, GetC0Size(tilingIns_->bType_.dataType));
    } else {
        return baseWidth;
    }
}

int32_t MatmulTilingAlgorithm::GetScaleABaseHeightAlign() const
{
    return MathUtil::Align(tilingIns_->tiling_.get_baseM(), GetC0Size(DataType::DT_FLOAT8_E8M0));
}

int32_t MatmulTilingAlgorithm::GetScaleABaseWidthAlign() const
{
    return MathUtil::CeilDivision(tilingIns_->tiling_.get_baseK(), SCALE_K_SIZE);
}

int32_t MatmulTilingAlgorithm::GetScaleBBaseHeightAlign() const
{
    return MathUtil::Align(MathUtil::CeilDivision(tilingIns_->tiling_.get_baseK(), SCALE_K_SIZE), GetC0Size(DataType::DT_FLOAT8_E8M0));
}

int32_t MatmulTilingAlgorithm::GetScaleBBaseWidthAlign() const
{
    if (!tilingIns_->bType_.isScaleTrans) {
        return MathUtil::Align(tilingIns_->tiling_.get_baseN(), GetC0Size(DataType::DT_FLOAT8_E8M0));
    } else {
        return tilingIns_->tiling_.get_baseN();
    }
}

int32_t MatmulTilingAlgorithm::GetMatrixAByteSize() const
{
    if (tilingIns_->aType_.pos == TPosition::VECOUT) {
        return MathUtil::Align(tilingIns_->tiling_.get_singleCoreM(), BLOCK_CUBE) *
            MathUtil::Align(tilingIns_->tiling_.get_singleCoreK(), C0_SIZE);
    } else if (tilingIns_->aType_.pos == TPosition::GM) {
        return GetABaseHeightAlign(tilingIns_->tiling_.get_baseM()) * GetABaseWidthAlign(tilingIns_->tiling_.get_baseK());
    } else {
        return 0;
    }
}

int32_t MatmulTilingAlgorithm::GetMatrixBByteSize() const
{
    if (tilingIns_->bType_.pos == TPosition::VECOUT) {
        return MathUtil::Align(tilingIns_->tiling_.get_singleCoreK(), BLOCK_CUBE) *
            MathUtil::Align(tilingIns_->tiling_.get_singleCoreN(), C0_SIZE);
    } else if (tilingIns_->bType_.pos == TPosition::GM) {
        return GetBBaseHeightAlign(tilingIns_->tiling_.get_baseK()) * GetBBaseWidthAlign(tilingIns_->tiling_.get_baseN());
    } else {
        return 0;
    }
}

int32_t MatmulTilingAlgorithm::GetMatrixScaleAByteSize() const
{
    if (tilingIns_->aType_.scalePos == TPosition::VECOUT) {
        return MathUtil::Align(tilingIns_->tiling_.get_singleCoreM(), BLOCK_CUBE) *
            MathUtil::Align(MathUtil::CeilDivision(tilingIns_->tiling_.get_singleCoreK(), SCALE_K_SIZE), C0_SIZE);
    } else if (tilingIns_->aType_.scalePos == TPosition::GM) {
        return GetScaleABaseHeightAlign() * GetScaleABaseWidthAlign();
    } else {
        return 0;
    }
}

int32_t MatmulTilingAlgorithm::GetMatrixScaleBByteSize() const
{
    if (tilingIns_->bType_.scalePos == TPosition::VECOUT) {
        return MathUtil::Align(MathUtil::CeilDivision(tilingIns_->tiling_.get_singleCoreK(), SCALE_K_SIZE), BLOCK_CUBE) *
            MathUtil::Align(tilingIns_->tiling_.get_singleCoreN(), C0_SIZE);
    } else if (tilingIns_->bType_.scalePos == TPosition::GM) {
        return GetScaleBBaseHeightAlign() * GetScaleBBaseWidthAlign();
    } else {
        return 0;
    }
}

void MatmulTilingAlgorithm::CalABAndScaleABL1Space(int32_t matrixByteSize, int32_t cacheNum, int32_t stepSize,
    uint32_t &curL1UpperHalfAddr, uint32_t &curL1LowerHalfAddr) const
{
    int32_t cacheFactor = (cacheNum / stepSize - 1) % NUM_TWO;
    int32_t queDepth = cacheFactor == 0 ? DB_OFF : DB_ON;

    uint32_t initBufferSize = static_cast<uint32_t>(matrixByteSize * stepSize);
    if (queDepth == DB_OFF) {
        curL1UpperHalfAddr += initBufferSize;
    } else if (queDepth == DB_ON) {
        curL1UpperHalfAddr += initBufferSize;
        curL1LowerHalfAddr += initBufferSize;
    }
}

bool MatmulTilingAlgorithm::EnableL1BankConflictOptimise() const
{
    // mdl
    if (tilingIns_->mmConfigType != 1) {
        TILING_LOG_WARNING("MatmulApi Tiling : L1BankConflictOptimise only support mdl");
        return false;
    }

    // gm in
    if (tilingIns_->aType_.pos != TPosition::GM || tilingIns_->bType_.pos != TPosition::GM ||
        (tilingIns_->aType_.hasSetScaleType && tilingIns_->aType_.scalePos != TPosition::GM) ||
        (tilingIns_->bType_.hasSetScaleType && tilingIns_->bType_.scalePos != TPosition::GM) ||
        (tilingIns_->isBias && tilingIns_->biasType_.pos != TPosition::GM)) {
        TILING_LOG_WARNING("MatmulApi Tiling : L1BankConflictOptimise only support gm in");
        return false;
    }

    /*
     * l1 space check, the rule must be same as init() in kernel
     */
    uint32_t curL1UpperHalfAddr = 0;
    uint32_t curL1LowerHalfAddr = tilingIns_->oriBufferPool_.l1Size / NUM_TWO;

    // A
    int32_t matrixByteSize = GetABaseHeightAlign(tilingIns_->tiling_.get_baseM()) * GetABaseWidthAlign(tilingIns_->tiling_.get_baseK()) * DTYPE_BIT_TAB.at(tilingIns_->aType_.dataType) / BITS_PER_BYTE;
    int32_t cacheNum = tilingIns_->tiling_.get_depthA1();
    int32_t stepSize = tilingIns_->tiling_.get_stepKa() * tilingIns_->tiling_.get_stepM();
    CalABAndScaleABL1Space(matrixByteSize, cacheNum, stepSize, curL1UpperHalfAddr, curL1LowerHalfAddr);

    // B
    matrixByteSize = GetBBaseHeightAlign(tilingIns_->tiling_.get_baseK()) * GetBBaseWidthAlign(tilingIns_->tiling_.get_baseN()) * DTYPE_BIT_TAB.at(tilingIns_->bType_.dataType) / BITS_PER_BYTE;
    cacheNum = tilingIns_->tiling_.get_depthB1();
    stepSize = tilingIns_->tiling_.get_stepKb() * tilingIns_->tiling_.get_stepN();
    CalABAndScaleABL1Space(matrixByteSize, cacheNum, stepSize, curL1UpperHalfAddr, curL1LowerHalfAddr);

    // ScaleA
    if (tilingIns_->aType_.hasSetScaleType) {
        matrixByteSize = GetScaleABaseHeightAlign() * GetScaleABaseWidthAlign() * DTYPE_BIT_TAB.at(DataType::DT_FLOAT8_E8M0) / BITS_PER_BYTE;
        cacheNum = tilingIns_->tiling_.get_depthA1();
        stepSize = tilingIns_->tiling_.get_stepKa() * tilingIns_->tiling_.get_stepM() *
            (static_cast<uint32_t>(tilingIns_->tiling_.get_mxTypePara()) & SCALE_FACTOR_A_MASK);
        CalABAndScaleABL1Space(matrixByteSize, cacheNum, stepSize, curL1UpperHalfAddr, curL1LowerHalfAddr);
    }

    // ScaleB
    if (tilingIns_->bType_.hasSetScaleType) {
        matrixByteSize = GetScaleBBaseHeightAlign() * GetScaleBBaseWidthAlign() * DTYPE_BIT_TAB.at(DataType::DT_FLOAT8_E8M0) / BITS_PER_BYTE;
        cacheNum = tilingIns_->tiling_.get_depthB1();
        stepSize = tilingIns_->tiling_.get_stepKb() * tilingIns_->tiling_.get_stepN() *
            ((static_cast<uint32_t>(tilingIns_->tiling_.get_mxTypePara()) & SCALE_FACTOR_B_MASK) >> SCALE_FACTOR_B_OFFSET);
        CalABAndScaleABL1Space(matrixByteSize, cacheNum, stepSize, curL1UpperHalfAddr, curL1LowerHalfAddr);
    }

    // bias
    if (tilingIns_->isBias) {
        curL1UpperHalfAddr += tilingIns_->tiling_.get_baseN() * sizeof(DTYPE_BYTE_TAB.at(tilingIns_->biasType_.dataType));
    }

    // quant
    if (tilingIns_->deqType == DequantType::TENSOR) {
        curL1LowerHalfAddr += tilingIns_->tiling_.get_baseN() * UINT64_TYPES;
    }

    if (curL1UpperHalfAddr > static_cast<uint32_t>(tilingIns_->oriBufferPool_.l1Size / NUM_TWO) ||
        curL1LowerHalfAddr > static_cast<uint32_t>(tilingIns_->oriBufferPool_.l1Size)) {
        TILING_LOG_WARNING("MatmulApi Tiling : Larger than upper or lower half of L1 buffer size. curL1UpperHalfAddr is %d, curL1LowerHalfAddr is %d.", curL1UpperHalfAddr, curL1LowerHalfAddr);
        return false;
    }

    return true;
}
} // namespace matmul_tiling