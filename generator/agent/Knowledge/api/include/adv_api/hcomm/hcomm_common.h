/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_common.h
 * \brief Hcomm common definitions
 */
#ifndef INCLUDE_ADV_API_HCOMM_HCOMM_COMMON_H
#define INCLUDE_ADV_API_HCOMM_HCOMM_COMMON_H

#include <cstdint>

namespace AscendC {
using HcommHandle = int32_t;
using ChannelHandle = uint64_t;

enum class CommEngine {
    AIV = 0,
    AICPU,
    CCU,
};

enum class CommProtocol {
    ROCE = 0,
    URMA,
    SDMA,
};

} // namespace AscendC

#endif // LIB_HCOMM_HCOMM_COMMON_H
