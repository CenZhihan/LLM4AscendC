/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "asrtc.h"

asrtcResult asrtcCreateProgram(asrtcProgram *prog, const char *src,
                                 const char *name, int numHeaders,
                                 const char *const *headers,
                                 const char *const *includeNames)
{
    return ASRTC_SUCCESS;
}

asrtcResult asrtcDestroyProgram(asrtcProgram *prog)
{
    return ASRTC_SUCCESS;
}

asrtcResult asrtcCompileProgram(asrtcProgram prog, int numOptions,
                                  const char *const *options)
{
    return ASRTC_SUCCESS;
}

asrtcResult asrtcGetDeviceELFSize(asrtcProgram prog,
                                    size_t *deviceELFSizeRet)
{
    return ASRTC_SUCCESS;
}

asrtcResult asrtcGetDeviceELF(asrtcProgram prog, char *deviceELF)
{
    return ASRTC_SUCCESS;
}

asrtcResult asrtcAddNameExpression(asrtcProgram prog,
                                     const char *const nameExpression)
{
    return ASRTC_SUCCESS;
}

asrtcResult bishengGetLoweredName(asrtcProgram prog,
                                  const char *nameExpression,
                                  const char **loweredName)
{
    return ASRTC_SUCCESS;
}

asrtcResult asrtcGetProgramLogSize(asrtcProgram prog, size_t *logSizeRet)
{
    return ASRTC_SUCCESS;
}

asrtcResult asrtcGetProgramLog(asrtcProgram prog, char *log)
{
    return ASRTC_SUCCESS;
}