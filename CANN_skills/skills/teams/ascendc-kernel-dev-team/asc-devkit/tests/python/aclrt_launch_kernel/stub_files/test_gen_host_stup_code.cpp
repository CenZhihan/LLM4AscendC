/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <securec.h>

#ifndef ASCENDC_DUMP
#define ASCENDC_DUMP 1
#endif

#if defined(ASCENDC_DUMP) && (ASCENDC_DUMP == 0)
    #undef ASCENDC_DUMP
#endif

static char ascendcErrMsg[1024] = {0};

static void *g_kernel_handle_aiv = nullptr;

struct ascend_kernels {
    uint32_t version;
    uint32_t type_cnt;
    uint32_t aiv_type;
    uint32_t aiv_len;
    uint32_t aiv_file_len;
    uint8_t aiv_buf[172464];
} __ascend_kernel_ascend910b1_ascendc_kernels_npu __attribute__ ((section (".ascend.kernel.ascend910b1.ascendc_kernels_npu"))) = {1,1,1,172464,172464,{0}};

extern "C" {
uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);
uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t numBlocks, void **args,
                            uint32_t size, const void *stream);
uint32_t GetAscendCoreSyncAddr(void **addr);
int UnregisterAscendBinary(void *hdl);
void StartAscendProf(const char *name, uint64_t *startTime);
void ReportAscendProf(const char *name, uint32_t numBlocks, uint32_t taskType, const uint64_t startTime);
bool GetAscendProfStatus();
uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);
uint32_t FreeAscendMemDevice(void *devMem);
bool AscendCheckSoCVersion(const char *socVersion, char* errMsg);
void AscendProfRegister();
uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum);
uint32_t LaunchAscendKernelForVectorCore(const char *opType, void *handle, const uint64_t key, void **args, uint32_t size,
    const void *stream, bool enableProf, uint32_t aicNumBlocks, uint32_t aivNumBlocks, uint32_t aivNumBlocksOffset);
}

namespace Adx {
    void AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,
                            void *stream, const char *opType);
}

    class KernelHandleGradUnregister {
    private:
        KernelHandleGradUnregister() {}

    public:
        KernelHandleGradUnregister(const KernelHandleGradUnregister&) = delete;
        KernelHandleGradUnregister& operator=(const KernelHandleGradUnregister&) = delete;

        static KernelHandleGradUnregister& GetInstance() {
            static KernelHandleGradUnregister instance;
            return instance;
        }
        ~KernelHandleGradUnregister(){
            if (g_kernel_handle_aiv) {
                UnregisterAscendBinary(g_kernel_handle_aiv);
                g_kernel_handle_aiv = nullptr;
            }
        }
    };

static void __register_kernels(void) __attribute__((constructor));
void __register_kernels(void)
{
    const char* compileSocVersion = "ascend910b1";
    uint32_t ret;

    bool checkSocVersion = AscendCheckSoCVersion(compileSocVersion, ascendcErrMsg);
    if (!checkSocVersion) {
        return;
    }
    ret = RegisterAscendBinary(
        (const char *)__ascend_kernel_ascend910b1_ascendc_kernels_npu.aiv_buf,
        __ascend_kernel_ascend910b1_ascendc_kernels_npu.aiv_file_len,
        1,
        &g_kernel_handle_aiv);
    if (ret != 0) {
        printf("RegisterAscendBinary aiv ret %u \n", ret);
    }

    AscendProfRegister();
}

struct A {
    uint16_t A_uint16_1;
    uint64_t A_uint64_1;
    bool A_bool_1;
    bool A_bool_2;
    uint16_t A_uint16_2;
};

struct AddCustomTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
};



uint32_t launch_and_profiling_add_custom(uint64_t func_key, uint32_t numBlocks, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "add_custom";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, numBlocks, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, numBlocks, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_add_custom(uint32_t numBlocks, void* stream, bool bool_1, bool bool_2, uint32_t uint32_1, A* tmpStruct, uint16_t uint16_1, void* x, void* y, void* z, AddCustomTilingData* tiling)
{
    struct {
    #if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
            void* __ascendc_dump;
    #endif
        alignas(((alignof(bool) + 3) >> 2) << 2) bool bool_1;
        alignas(((alignof(bool) + 3) >> 2) << 2) bool bool_2;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t uint32_1;
        alignas(((alignof(A) + 3) >> 2) << 2) A tmpStruct;
        alignas(((alignof(uint16_t) + 3) >> 2) << 2) uint16_t uint16_1;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* x;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* y;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* z;
        alignas(((alignof(AddCustomTilingData) + 3) >> 2) << 2) AddCustomTilingData tiling;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    constexpr uint32_t __ascendc_one_core_dump_size = 1048576;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_dump), __ascendc_one_core_dump_size * 75);
#endif
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.bool_1 = bool_1;
    __ascendc_args.bool_2 = bool_2;
    __ascendc_args.uint32_1 = uint32_1;
    (void) memcpy_s(&__ascendc_args.tmpStruct, sizeof(__ascendc_args.tmpStruct), tmpStruct, sizeof(__ascendc_args.tmpStruct));
    __ascendc_args.uint16_1 = uint16_1;
    __ascendc_args.x = x;
    __ascendc_args.y = y;
    __ascendc_args.z = z;
    (void) memcpy_s(&__ascendc_args.tiling, sizeof(__ascendc_args.tiling), tiling, sizeof(__ascendc_args.tiling));

    const char *__ascendc_name = "add_custom";
    __ascendc_ret = launch_and_profiling_add_custom(0, numBlocks, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    Adx::AdumpPrintWorkSpace(__ascendc_args.__ascendc_dump, __ascendc_one_core_dump_size * 75, stream, __ascendc_name);
    FreeAscendMemDevice(__ascendc_args.__ascendc_dump);
#endif
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}
