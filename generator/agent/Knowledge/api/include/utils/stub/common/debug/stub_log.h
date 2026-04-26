/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef COMMON_DEBUG_STUB_LOG_H
#define COMMON_DEBUG_STUB_LOG_H

#include <sys/syscall.h>
#include <unistd.h>
#include <inttypes.h>
#include "dlog_pub.h"

#define GE_MODULE_NAME static_cast<int32_t>(GE)

#if !(defined(UT_TEST) || defined(ST_TEST))
#define GELOGE(ERROR_CODE, fmt, ...)                                                                \
    do {                                                                                              \
        dlog_error(GE_MODULE_NAME, "%" PRIu64 " %s: ErrorNo: %" PRIuLEAST8 "(%s) " fmt,                 \
        syscall(SYS_gettid), &__FUNCTION__[0U],                                                  \
            (ERROR_CODE), ("Ascend Log Tiling Sink"), ##__VA_ARGS__);            \
    } while (false)

#define GELOGW(fmt, ...)                                                                          \
    do {                                                                                            \
        dlog_warn(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, syscall(SYS_gettid), &__FUNCTION__[0U], ##__VA_ARGS__); \
    } while (false)

#define GELOGI(fmt, ...)                                                                          \
    do {                                                                                            \
        dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, syscall(SYS_gettid), &__FUNCTION__[0U], ##__VA_ARGS__); \
    } while (false)

#define GELOGD(fmt, ...)                                                                           \
    do {                                                                                             \
        dlog_debug(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, syscall(SYS_gettid), &__FUNCTION__[0U], ##__VA_ARGS__); \
    } while (false)
#else
#define GELOGE
#define GELOGW
#define GELOGI
#define GELOGD
#endif

#endif  // COMMON_DEBUG_STUB_LOG_H_

