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
 * \file conv3d_tiling.cpp
 * \brief
 */

#include <cstdint>
#include "conv3d_tiling_algorithm.h"
#include "../../detail/host_log.h"
#include "include/adv_api/conv/conv3d/conv3d_tiling_base.h"
#include "include/adv_api/conv/conv3d/conv3d_tiling.h"

namespace Conv3dTilingApi {
int64_t Conv3dTiling::GetTiling(optiling::TConv3DApiTiling &tiling)
{
    int64_t ret = Compute();
    if (ret == -1) {
        TILING_LOG_ERROR("can not gen conv3d api tiling");
        return -1;
    }

    SetFinalTiling(tiling);
    PrintTilingData();
    return ret;
}

int64_t Conv3dTiling::GetTiling(AscendC::tiling::TConv3DApiTiling& tiling)
{
    int64_t ret1 = Compute();
    if (ret1 == -1) {
        TILING_LOG_ERROR("can not gen conv3d api tiling");
        return -1;
    }

    SetFinalTiling(tiling);
    PrintTilingData();
    return ret1;
}

int64_t Conv3dTiling::Compute()
{
    if (!CheckSocVersion()) {
        return -1;
    }
    if (!CheckInputParam()) {
        return -1;
    }
    // get cube info
    GetCubeInfo();
    // cal output and check valid
    if (!ShapeInitCalc()) {
        return -1;
    }
    if (!CheckParamsOverflow()) {
        return -1;
    }
    Conv3dTilingAlgorithm tilingAlgo(this);
    int64_t ret = tilingAlgo.Process();
    return ret;
}
} // namespace Conv3dTilingApi
