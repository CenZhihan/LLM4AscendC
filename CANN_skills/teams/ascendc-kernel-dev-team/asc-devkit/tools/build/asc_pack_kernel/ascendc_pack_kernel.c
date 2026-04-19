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
 * \file ascendc_pack_kernel.c
 * \brief
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include "ascendc_pack_kernel.h"

size_t GetFileSize(const char* filePath)
{
    FILE *file = fopen(filePath, "rb");
    if (file == NULL) {
        printf("[Error] open file: %s failed.\n", filePath);
        return 0;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        printf("[Error] fseek file: %s failed.\n", filePath);
        (void)fclose(file);
        return 0;
    }

    size_t size = (size_t)ftell(file);
    (void)fclose(file);
    return size;
}

size_t ReadFile(const char *file, void *buf, size_t len)
{
    int fd = open(file, O_RDONLY);
    size_t size = (size_t)read(fd, buf, len);
    if (size > len) {
        printf("[Error] read %s size > len, %lu > %lu\n", file, size, len);
    }
    (void)close(fd);
    return size;
}

size_t WriteFile(const char *file, void *buf, size_t len)
{
    int fd = open(file, O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR);
    size_t size = (size_t)write(fd, buf, len);
    if (size > len) {
        printf("[Error] write %s size > len, %lu > %lu\n", file, size, len);
    }
    (void)close(fd);
    return size;
}

#if !(defined(UT_TEST) || defined(ST_TEST))
int main(int argc, char *argv[])
#else
int AscendcPackKernelMain(int argc, char *argv[])
#endif
{
    if (argc != 0x5) {
        printf("[Error] %s <elf_in> <elf_add> <kernel_type> <elf_out>\n", argv[0]);
        return 1;
    }

    const char* srcFile = argv[1];
    const char* kernelFile = argv[2];
    const char* kernelType = argv[3];
    const char* dstFile = argv[4];

    size_t srcFileSize = GetFileSize(srcFile);
    size_t kernelFileSize = GetFileSize(kernelFile);
    if ((srcFileSize == 0) || kernelFileSize == 0) {
        return 1;
    }

    uint8_t *src = (uint8_t *)malloc(srcFileSize);
    CHECK_COND_AND_DO((src == NULL), {
        printf("[Error] malloc src failed!\n");
        return 1;
    });

    uint8_t *dst = (uint8_t *)malloc(srcFileSize);
    CHECK_COND_AND_DO((dst == NULL), {
        printf("[Error] malloc dst failed!\n");
        free(src);
        return 1;
    });

    // read kernel file to copy kernel to section
    uint8_t *sec = (uint8_t *)malloc(kernelFileSize);
    CHECK_COND_AND_DO((sec == NULL), {
        printf("[Error] malloc sec failed!\n");
        free(src);
        free(dst);
        return 1;
    });

    (void)memset_s(dst, srcFileSize, 0, srcFileSize);

    size_t elfAddLen = ReadFile(kernelFile, sec, kernelFileSize);
    size_t ssz = ReadFile(srcFile, src, srcFileSize);
    uint32_t type = (uint32_t)strtol(kernelType, NULL, 10);
    if (type >= ELF_TYPE_MAX) {
        printf("[Error] sec_name type: %s is error!\n", kernelType);
        free(src);
        free(dst);
        free(sec);
        return 1;
    }

    size_t dsz = ElfAddSection(src, ssz, dst, srcFileSize, sec, elfAddLen, type);
    (void)WriteFile(dstFile, dst, dsz);
    free(src);
    free(dst);
    free(sec);
    return 0;
}
