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
 * \file stub_util.h
 * \brief
 */

#ifndef COMMON_DEBUG_STUB_UTIL_H
#define COMMON_DEBUG_STUB_UTIL_H

#include <memory>
#include <utility>
#include "common/debug/stub_log.h"
#include "graph/ge_error_codes.h"

namespace ge {

template<typename T, typename... Args>
static inline std::shared_ptr<T> ComGraphMakeShared(Args &&...args)
{
    using Tn = typename std::remove_const<T>::type;
    std::shared_ptr<T> ret = nullptr;
    try {
        ret = std::make_shared<Tn>(std::forward<Args>(args)...);
    } catch (const std::bad_alloc &) {
        ret = nullptr;
        GELOGE(GRAPH_FAILED, "Make shared failed");
    }
    return ret;
}

}  // namespace ge
#endif  // COMMON_DEBUG_STUB_UTIL_H_
