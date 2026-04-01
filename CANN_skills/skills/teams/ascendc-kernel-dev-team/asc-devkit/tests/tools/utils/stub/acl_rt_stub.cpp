/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "acl/acl_rt.h"

int aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    return 0;
}

int aclrtRegisterCpuFunc(const aclrtBinHandle handle, const char *funcName,
    const char *kernelName, aclrtFuncHandle *funcHandle)
{
    return 0;
}

int aclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle *argsHandle)
{
    return 0;
}

int aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void *param, size_t paramSize,
    aclrtParamHandle *paramHandle)
{
    return 0;
}

int aclrtKernelArgsFinalize(aclrtArgsHandle argsHandl)
{
    return 0;
}

int aclrtGetDevice(int32_t* deviceId)
{
    return 0;
}

int aclrtSetDevice(int32_t deviceId)
{
    return 0;
}

int aclrtResetDevice(int32_t deviceId)
{
    return 0;
}


int aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    return 0;
}

int aclrtGetFunctionName(aclrtFuncHandle funcHandle, uint32_t maxLen, char *name)
{
    return 0;
}

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream)
{
    return ACL_ERROR_NONE;
}

aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event)
{
    return ACL_ERROR_NONE;
}

aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream)
{
    return ACL_ERROR_NONE;
}

aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value)
{
    return ACL_ERROR_NONE;
}

aclError aclrtDestroyEvent(aclrtEvent event)
{
    return ACL_ERROR_NONE;
}

const char *aclrtGetSocName()
{
    return " ";
}

aclError aclrtDestroyStream(aclrtStream stream)
{
    return ACL_ERROR_NONE;
}

aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag)
{
    return ACL_ERROR_NONE;
}

aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag)
{
    return ACL_ERROR_NONE;
}

aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag)
{
    return ACL_ERROR_NONE;
}

aclError aclrtFree(void *devPtr)
{
    if (devPtr == nullptr) {
        return 1;
    }
    return ACL_ERROR_NONE;
}

aclError aclrtMallocWithCfg(void **devPtr, size_t size, aclrtMemMallocPolicy policy,
    aclrtMallocConfig *cfg)
{
    if (devPtr == nullptr) {
        return 1;
    }
    return ACL_ERROR_NONE;
}

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char *kernelName, aclrtFuncHandle *funcHandle)
{
    return 0;
}

aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t numBlocks, aclrtStream stream,
    aclrtLaunchKernelCfg *cfg, void *hostArgs, size_t argsSize, aclrtPlaceHolderInfo *placeHolderArray,
    size_t placeHolderNum)
{
    return 0;
}

aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode)
{
    return 0;
}