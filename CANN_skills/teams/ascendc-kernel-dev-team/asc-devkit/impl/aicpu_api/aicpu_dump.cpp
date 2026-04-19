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
 * \file aicpu_dump.cpp
 * \brief
 */

#include <stdarg.h>
#include <mutex>
#include <stdio.h>
#include "securec.h"
#include "aicpu_api/aicpu_api.h"

struct DumpConfig {
    void *dumpBufAddr;
    size_t dumpBufSize;
    size_t dumpOffset;
};

std::mutex dump_mutex;
extern "C" {
DumpConfig g_aicpuDumpConfig = {reinterpret_cast<void*>(0x8), 0x100000, 0x0};
}

namespace AscendC {
void printf(const char *fmt, ...)
{
    std::lock_guard<std::mutex> lock(dump_mutex);
    void *aicpuDumpBuffer = g_aicpuDumpConfig.dumpBufAddr;
    size_t aicpuDumpBufSize = g_aicpuDumpConfig.dumpBufSize;
    size_t aicpuDumpOffSet = 0;
    if (aicpuDumpBuffer) {
        aicpuDumpOffSet = *reinterpret_cast<size_t*>(aicpuDumpBuffer);
    } else {
        return;
    }
    if (aicpuDumpOffSet >= aicpuDumpBufSize) {
        return;
    }
    va_list otherArgs;
    va_start(otherArgs, fmt);
    aicpuDumpOffSet += vsnprintf_s(reinterpret_cast<char *>(aicpuDumpBuffer) + aicpuDumpOffSet,
        aicpuDumpBufSize - aicpuDumpOffSet, aicpuDumpBufSize - aicpuDumpOffSet, fmt, otherArgs);
    *reinterpret_cast<size_t*>(g_aicpuDumpConfig.dumpBufAddr) = aicpuDumpOffSet;
    va_end(otherArgs);
}
}
