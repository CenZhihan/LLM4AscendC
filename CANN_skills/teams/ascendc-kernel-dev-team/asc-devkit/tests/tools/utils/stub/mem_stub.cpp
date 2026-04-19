/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "mem.h"
#include "runtime/context.h"
#include "runtime/base.h"
#include "runtime/kernel.h"
#include <string>
#include <vector>

rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId)
{
    if (devPtr == nullptr) {
        return 1;
    }
    return RT_ERROR_NONE;
}

rtError_t rtFree(void *devPtr)
{
    if (devPtr == nullptr) {
        return 1;
    }
    return RT_ERROR_NONE;
}

rtError_t rtStreamSynchronizeWithTimeout(rtStream_t stm, int32_t timeout)
{
    return RT_ERROR_NONE;
}

rtError_t rtStreamCreateWithFlags(rtStream_t *stream, int32_t priority, uint32_t flags)
{
    return RT_ERROR_NONE;
}

rtError_t rtEventCreateWithFlag(rtEvent_t *event_, uint32_t flag)
{
    return RT_ERROR_NONE;
}

rtError_t rtStreamDestroy(rtStream_t stream)
{
    return RT_ERROR_NONE;
}

rtError_t rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t numBlocks, rtArgsEx_t *argsInfo,
    rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
{
    return RT_ERROR_NONE;
}

rtError_t rtVectorCoreKernelLaunchWithHandle(void *hdl, const uint64_t tilingKey, uint32_t numBlocks,
    rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
{
    return RT_ERROR_NONE;
}

rtError_t rtCtxGetCurrent(rtContext_t *ctx)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetDevice(int32_t *device)
{
    return RT_ERROR_NONE;
}

rtError_t rtEventRecord(rtEvent_t event, rtStream_t stream)
{
    return RT_ERROR_NONE;
}

rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event)
{
    return RT_ERROR_NONE;
}


rtError_t rtEventReset(rtEvent_t event, rtStream_t stream)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t *val)
{
    return RT_ERROR_NONE;
}

rtError_t rtEventDestroy(rtEvent_t event)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetSocVersion(char *version, const uint32_t maxLen)
{
    return 0;
}

struct DumpBlacklist {
    std::string name;
    std::vector<std::string> pos;
};

struct ModelDumpConfig {
    std::string model_name;
    std::vector<std::string> layers;
    std::vector<std::string> watcher_nodes;
    std::vector<DumpBlacklist> optype_blacklist;
    std::vector<DumpBlacklist> opname_blacklist;
    std::vector<std::pair<std::string, std::string>> dump_op_ranges;
};

struct DumpConfig {
    std::string dump_path;
    std::string dump_mode;
    std::string dump_status;
    std::string dump_op_switch;
    std::string dump_debug;
    std::string dump_step;
    std::string dump_exception;
    std::vector<ModelDumpConfig> dump_list;
    std::string dump_data;
    std::string dump_level;
    std::vector<std::string> dump_stats;
};

namespace ge {
enum class MngActionType : uint32_t {
    DESTROY_STREAM,
    DESTROY_CONTEXT,
    RESET_DEVICE,
};

typedef union {
    rtStream_t stream;
    rtContext_t context;
    int32_t device_id;
} MngResourceHandle;

enum class StreamMngFuncType : uint32_t {
    ACLNN_STREAM_CALLBACK, // aclnn callback function for destroying sub-stream
};

using StreamMngFunc = uint32_t (*)(MngActionType action_type, MngResourceHandle handle);

class StreamMngFuncRegister{
    StreamMngFuncRegister(const StreamMngFuncType func_type, StreamMngFunc const manage_func) {}
};
}