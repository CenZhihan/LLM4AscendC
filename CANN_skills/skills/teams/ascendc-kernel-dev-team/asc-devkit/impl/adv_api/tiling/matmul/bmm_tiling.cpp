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
 * \file bmm_tiling.cpp
 * \brief
 */
#include <iostream>
#include <algorithm>
#include "math_util.h"
#include "../../detail/host_log.h"
#include "matmul_tiling_algorithm.h"
#include "include/adv_api/matmul/bmm_tiling.h"

using namespace std;
namespace matmul_tiling {
int32_t MultiCoreMatmulTiling::SetDim(int32_t dim)
{
    this->blockDim = dim;
    return 0;
}

int32_t MultiCoreMatmulTiling::SetSingleShape(int32_t singleMIn, int32_t singleNIn, int32_t singleKIn)
{
    TILING_LOG_DEBUG(" Set SingleCore M: %d", singleMIn);
    TILING_LOG_DEBUG(" Set SingleCore N: %d", singleNIn);
    TILING_LOG_DEBUG(" Set SingleCore K: %d", singleKIn);
    if (singleMIn > 0) {
        this->singleCoreM = singleMIn;
    }
    if (singleNIn > 0) {
        this->singleCoreN = singleNIn;
    }
    if (singleKIn > 0) {
        this->singleCoreK = singleKIn;
    }

    return 0;
}

int32_t MultiCoreMatmulTiling::SetAlignSplit(int32_t alignM, int32_t alignN, int32_t alignK)
{
    if (alignM > 0) {
        this->alignSingleM = alignM;
    }
    if (alignN > 0) {
        this->alignSingleN = alignN;
    }
    if (alignK > 0) {
        this->alignSingleK = alignK;
    }
    return 0;
}

int32_t MultiCoreMatmulTiling::SetShape(int32_t m, int32_t n, int32_t k)
{
    TILING_LOG_DEBUG(" Set Single M: %d", m);
    TILING_LOG_DEBUG(" Set Single N: %d", n);
    TILING_LOG_DEBUG(" Set Single K: %d", k);
    this->singleM = m;
    this->singleN = n;
    this->singleK = k;
    if (this->orgM == -1 && this->orgN == -1 && this->orgKa == -1 && this->orgKb == -1) {
        this->orgM = m != -1 ? m : this->orgM;
        this->orgN = n != -1 ? n : this->orgN;
        this->orgKa = k != -1 ? k : this->orgKa;
        this->orgKb = k != -1 ? k : this->orgKb;
    }
    return 0;
}

int64_t MultiCoreMatmulTiling::GetTiling(optiling::TCubeTiling& tiling)
{
    int32_t ret = Compute();
    if (ret == -1) {
        TILING_LOG_INFO("Cannot deduce tiling params from given info.");
        return ret;
    }
    SetFinalTiling(tiling);
    PrintTilingDataInfo(tiling);
    return ret;
}

int64_t MultiCoreMatmulTiling::GetTiling(AscendC::tiling::TCubeTiling& tiling)
{
    optiling::TCubeTiling mmTiling;
    int64_t ret = GetTiling(mmTiling);
    mmTiling.SaveToBuffer(&tiling, sizeof(TCubeTiling));
    return ret;
}

int64_t MultiCoreMatmulTiling::Compute()
{
    if (!CheckSetParam()) {
        TILING_LOG_INFO("Tiling compute params check don't pass.");
        return -1;
    }
    MatmulTilingAlgorithm algoIns(this);
    int ret = algoIns.Process();
    return ret;
}

int32_t MultiCoreMatmulTiling::GetSingleShape(int32_t& shapeM, int32_t& shapeN, int32_t& shapeK)
{
    shapeM = this->tiling_.get_singleCoreM();
    shapeN = this->tiling_.get_singleCoreN();
    shapeK = this->tiling_.get_singleCoreK();

    return 0;
}

int32_t MultiCoreMatmulTiling::GetCoreNum(int32_t& dim, int32_t& mDim, int32_t& nDim)
{
    dim = this->tiling_.get_usedCoreNum();
    mDim = MathUtil::CeilDivision(this->singleM, this->tiling_.get_singleCoreM());
    nDim = MathUtil::CeilDivision(this->singleN, this->tiling_.get_singleCoreN());
    return 0;
}

bool MultiCoreMatmulTiling::EnableL1BankConflictOptimise()
{
    MatmulTilingAlgorithm algoIns(this);
    bool ret = algoIns.EnableL1BankConflictOptimise();
    return ret;
}

int32_t BatchMatmulTiling::SetBatch(int32_t batchMIn, int32_t batchNIn)
{
    this->batchM = batchMIn;
    this->batchN = batchNIn;

    return 0;
}
int32_t BatchMatmulTiling::SetSingleBatch(int32_t singleMIn, int32_t singleNIn)
{
    if (singleMIn > 0) {
        this->singleBatchM = singleMIn;
    }
    if (singleNIn > 0) {
        this->singleBatchN = singleNIn;
    }
    return 0;
}

int32_t BatchMatmulTiling::GetCoreNum(int32_t& dim, int32_t& mDim, int32_t& nDim, int32_t& batchCoreM, int32_t& batchCoreN)
{
    dim = this->tiling_.get_usedCoreNum();
    mDim = MathUtil::CeilDivision(this->tiling_.get_M(), this->tiling_.get_singleCoreM());
    nDim = MathUtil::CeilDivision(this->tiling_.get_N(), this->tiling_.get_singleCoreN());
    batchCoreM = MathUtil::CeilDivision(this->batchM, this->singleBatchM);
    batchCoreN = MathUtil::CeilDivision(this->batchN, this->singleBatchN);
    return 0;
}

int64_t BatchMatmulTiling::GetTiling(optiling::TCubeTiling& tiling)
{
    int32_t ret = Compute();
    if (ret == -1) {
        TILING_LOG_INFO("Cannot deduce tiling params from given info.");
        return -1;
    }
    SetFinalTiling(tiling);
    PrintTilingDataInfo(tiling);
    return 0;
}

int64_t BatchMatmulTiling::GetTiling(AscendC::tiling::TCubeTiling& tiling)
{
    optiling::TCubeTiling mmTiling;
    int64_t ret = GetTiling(mmTiling);
    mmTiling.SaveToBuffer(&tiling, sizeof(TCubeTiling));
    return ret;
}

int64_t BatchMatmulTiling::Compute()
{
    if (!CheckSetParam()) {
        TILING_LOG_INFO("Tiling compute params check don't pass.");
        return -1;
    }
    MatmulTilingAlgorithm algoIns(this);
    int ret = algoIns.Process();
    return ret;
}
} // namespace matmul_tiling

extern "C" {
int32_t MultiCoreMatmulGetTmpBufSize(optiling::TCubeTiling& tiling, matmul_tiling::SysTilingTempBufSize& bufSize)
{
    bufSize.l1Size = max(bufSize.l1Size, tiling.get_shareL1Size());
    bufSize.l0cSize = max(bufSize.l0cSize, tiling.get_shareL0CSize());
    // V200 ND2NZ needs 2 * tiling.get_transLength() UB buffer
    bufSize.ubSize = max(bufSize.ubSize, max(tiling.get_shareUbSize(), 2 * tiling.get_transLength()));
    return 0;
}
int32_t BatchMatmulGetTmpBufSize(optiling::TCubeTiling& tiling, matmul_tiling::SysTilingTempBufSize& bufSize)
{
    bufSize.l1Size = max(bufSize.l1Size, tiling.get_shareL1Size());
    bufSize.l0cSize = max(bufSize.l0cSize, tiling.get_shareL0CSize());
    // V200 ND2NZ needs 2 * tiling.get_transLength() UB buffer
    bufSize.ubSize = max(bufSize.ubSize, max(tiling.get_shareUbSize(), 2 * tiling.get_transLength()));
    return 0;
}
int32_t MultiCoreMatmulGetTmpBufSizeV2(AscendC::tiling::TCubeTiling& tiling, matmul_tiling::SysTilingTempBufSize& bufSize)
{
    bufSize.l1Size = max(bufSize.l1Size, tiling.shareL1Size);
    bufSize.l0cSize = max(bufSize.l0cSize, tiling.shareL0CSize);
    // V200 ND2NZ needs 2 * tiling.get_transLength() UB buffer
    bufSize.ubSize = max(bufSize.ubSize, max(tiling.shareUbSize, 2 * tiling.transLength));
    return 0;
}
int32_t BatchMatmulGetTmpBufSizeV2(AscendC::tiling::TCubeTiling& tiling, matmul_tiling::SysTilingTempBufSize& bufSize)
{
    bufSize.l1Size = max(bufSize.l1Size, tiling.shareL1Size);
    bufSize.l0cSize = max(bufSize.l0cSize, tiling.shareL0CSize);
    // V200 ND2NZ needs 2 * tiling.get_transLength() UB buffer
    bufSize.ubSize = max(bufSize.ubSize, max(tiling.shareUbSize, 2 * tiling.transLength));
    return 0;
}
};
