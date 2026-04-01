/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "kernel.h"
#include "acl/acl_base.h"
#include "acl/acl_rt.h"

rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **hdl)
{
    return 0;
}

aclError aclrtBinaryLoadFromData(const void *data, size_t length,
    const aclrtBinaryLoadOptions *options, aclrtBinHandle *binHandle)
{
    return 0;
}

rtError_t rtDevBinaryUnRegister(void *hdl)
{
    return 0;
}

aclError aclrtBinaryUnLoad(aclrtBinHandle binHandle)
{
    return 0;
}

rtError_t rtKernelLaunchWithHandle(void *hdl, const uint64_t tilingKey, uint32_t numBlocks,
                                           rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm,
                                           const void *kernelInfo)
{
    return 0;
}

aclError aclrtLaunchKernelWithConfig(aclrtFuncHandle funcHandle, uint32_t numBlocks,
                                                         aclrtStream stream, aclrtLaunchKernelCfg *cfg,
                                                         aclrtArgsHandle argsHandle, void *reserve)
{
    return 0;
}

rtError_t rtCtxGetCurrentDefaultStream(rtStream_t *stm)
{
    return 0;
}

aclError aclrtCtxGetCurrentDefaultStream(aclrtStream *stream)
{
    return 0;
}

rtError_t rtRegStreamStateCallback(const char_t *regName, rtStreamStateCallback callback)
{
    return 0;
}

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl)
{
    return 0;
}

rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char_t *stubName, const void *kernelInfoExt,
                             uint32_t funcMode)
{
    return 0;
}

rtError_t rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t numBlocks, rtArgsEx_t *argsInfo,
                                   rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo)
{
    return 0;
}

rtError_t rtFunctionGetMetaInfo(const rtFuncHandle funcHandle, const rtFunctionMetaType type, void *data,
    const uint32_t length)
{
    return 0;
}