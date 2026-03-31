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
 * \file matmul_tiling.cpp
 * \brief
 */

#include "include/adv_api/matmul/matmul_tiling.h"

#include <iostream>
#include <map>
#include <algorithm>

#include "../../detail/host_log.h"
#include "math_util.h"
#include "matmul_tiling_algorithm.h"

using namespace std;
namespace matmul_tiling {
int64_t MatmulApiTiling::GetTiling(optiling::TCubeTiling &tiling)
{
    const int64_t ret = Compute();
    if (ret == -1) {
        TILING_LOG_INFO("Cannot deduce tiling params from given info.");
        return -1;
    }
    SetFinalTiling(tiling);
    PrintTilingDataInfo(tiling);
    return ret;
}
int64_t MatmulApiTiling::GetTiling(AscendC::tiling::TCubeTiling &tiling)
{
    const int64_t ret = Compute();
    if (ret == -1) {
        TILING_LOG_INFO("Cannot deduce tiling params from given info.");
        return -1;
    }
    SetFinalTiling(tiling);
    PrintTilingDataInfo(tiling);
    return ret;
}

int64_t MatmulApiTiling::Compute()
{
    if (!CheckSetParam()) {
        TILING_LOG_INFO("Tiling compute params check don't pass.");
        return -1;
    }

    MatmulTilingAlgorithm algoIns(this);
    const int64_t ret = algoIns.Process();
    return ret;
}

bool MatmulApiTiling::EnableL1BankConflictOptimise()
{
    MatmulTilingAlgorithm algoIns(this);
    bool ret = algoIns.EnableL1BankConflictOptimise();
    return ret;
}
} // namespace matmul_tiling

extern "C" {
// bufSize used size
int32_t MatmulGetTmpBufSize(optiling::TCubeTiling &tiling, matmul_tiling::SysTilingTempBufSize &bufSize)
{
    bufSize.l1Size = max(bufSize.l1Size, tiling.get_shareL1Size());
    bufSize.l0cSize = max(bufSize.l0cSize, tiling.get_shareL0CSize());
    // V200 ND2NZ needs 2 * tiling.get_transLength() UB buffer
    bufSize.ubSize = max(bufSize.ubSize, max(tiling.get_shareUbSize(), 2 * tiling.get_transLength()));
    return 0;
}

// bufSize used size
int32_t MatmulGetTmpBufSizeV2(TCubeTiling &tiling, matmul_tiling::SysTilingTempBufSize &bufSize)
{
    bufSize.l1Size = max(bufSize.l1Size, tiling.shareL1Size);
    bufSize.l0cSize = max(bufSize.l0cSize, tiling.shareL0CSize);
    // V200 ND2NZ needs 2 * tiling.get_transLength() UB buffer
    bufSize.ubSize = max(bufSize.ubSize, max(tiling.shareUbSize, 2 * tiling.transLength));
    return 0;
}
}; // namespace matmul_tiling
