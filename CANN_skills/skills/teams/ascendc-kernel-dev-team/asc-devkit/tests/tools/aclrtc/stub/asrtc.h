/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef STUB_ACL_RT_COMPILE_H
#define STUB_ACL_RT_COMPILE_H

#include <stddef.h>

typedef void *asrtcProgram;

typedef enum {
    ASRTC_SUCCESS = 0,
    ASRTC_ERROR_OUT_OF_MEMORY,
    ASRTC_ERROR_PROGRAM_CREATION_FAILURE,
    ASRTC_ERROR_INVALID_PROGRAM,
    ASRTC_ERROR_INVALID_INPUT,
    ASRTC_ERROR_INVALID_OPTION,
    ASRTC_ERROR_COMPILE,
    ASRTC_ERROR_LINK,
    ASRTC_ERROR_NOT_IMPLEMENTED,
    ASRTC_ERROR_INTERNAL_ERROR,
    ASRTC_ERROR_IO,
    ASRTC_ERROR_UNKNOWN
} asrtcResult;

asrtcResult asrtcCreateProgram(asrtcProgram *prog, const char *src,
                                 const char *name, int numHeaders,
                                 const char *const *headers,
                                 const char *const *includeNames);

asrtcResult asrtcDestroyProgram(asrtcProgram *prog);

asrtcResult asrtcCompileProgram(asrtcProgram prog, int numOptions,
                                  const char *const *options);

asrtcResult asrtcGetDeviceELFSize(asrtcProgram prog,
                                    size_t *deviceELFSizeRet);

asrtcResult asrtcGetDeviceELF(asrtcProgram prog, char *deviceELF);

asrtcResult asrtcAddNameExpression(asrtcProgram prog,
                                     const char *const nameExpression);

asrtcResult bishengGetLoweredName(asrtcProgram prog,
                                  const char *nameExpression,
                                  const char **loweredName);

asrtcResult asrtcGetProgramLogSize(asrtcProgram prog, size_t *logSizeRet);

asrtcResult asrtcGetProgramLog(asrtcProgram prog, char *log);

#endif // STUB_ACL_RT_COMPILE_H