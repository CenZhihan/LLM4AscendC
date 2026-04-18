/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>

#undef __CHECK_FEATURE_AT_PRECOMPILE
#define ASCENDC_DUMP 1
#define __simt_callee__ 
#define __gm__ 
#define __aicore__ 

struct bfloat16_t
{
    uint16_t value;
};

struct half
{
    uint16_t value;
};

enum L1CacheType : uint32_t { NON_CACHEABLE = 0, CACHEABLE = 1 };
enum class LD_L2CacheType : uint32_t { L2_CACHE_HINT_NORMAL_FV = 0 };
enum class ST_L2CacheType : uint32_t { L2_CACHE_HINT_NORMAL_FV = 0 };

template <LD_L2CacheType L2Cache = LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
          L1CacheType L1CacheType = L1CacheType::NON_CACHEABLE, typename T>
T __ldg(__gm__ T* address)
{
    return *address;
}

struct dim3
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
};
namespace __asc_simt_vf {
static dim3 blockIdx = {8, 0, 0};
static dim3 threadIdx = {512, 0, 0};
}
uint64_t atomicAdd(uint64_t *addr, uint64_t val);

void __threadfence();

void __sync_workitems();

namespace __cce_scalar {
void dcci(uint8_t *addr, uint64_t entire, uint64_t type)
{
    return;
}
}

#undef ASCENDC_CPU_DEBUG
#include "impl/simt_api/cpp/kernel_simt_print_intf_impl.h"

class PrintfTestsuite : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

struct PrintTlvCheck
{
    uint32_t type = 0U;
    uint32_t length = 0U;
    uint32_t blockIdx[3] = {0U};
    uint32_t threadIdx[3] = {0U};
    uint32_t resv[4];
    uint64_t fmtOffset = 0U;
    uint64_t param[2];
    char fmt[16];
    char str[5];
};

TEST_F(PrintfTestsuite, CPPPrintfTest)
{
    uint32_t bufSize = 1024;
    g_sysSimtPrintFifoSpace = reinterpret_cast<uint8_t*>(malloc(bufSize));
    ASSERT_NE(nullptr, g_sysSimtPrintFifoSpace);
    __asc_simt_vf::BlockRingBufInfo* blockInfo = reinterpret_cast<__asc_simt_vf::BlockRingBufInfo*>(g_sysSimtPrintFifoSpace);
    blockInfo->length = bufSize;
    blockInfo->ringBufLen = bufSize - sizeof(__asc_simt_vf::BlockRingBufInfo) - sizeof(__asc_simt_vf::RingBufWriteInfo) - sizeof(__asc_simt_vf::RingBufReadInfo);
    blockInfo->magic = __asc_simt_vf::MAGIC;
    blockInfo->flag = 1;
    blockInfo->ringBufAddr = reinterpret_cast<uint64_t>(g_sysSimtPrintFifoSpace) + sizeof(__asc_simt_vf::BlockRingBufInfo) + sizeof(__asc_simt_vf::RingBufWriteInfo);

    __asc_simt_vf::RingBufReadInfo* readInfo = reinterpret_cast<__asc_simt_vf::RingBufReadInfo*>(g_sysSimtPrintFifoSpace + sizeof(__asc_simt_vf::BlockRingBufInfo));
    readInfo->type = static_cast<uint32_t>(__asc_simt_vf::DumpType::DUMP_BUFO);
    readInfo->length = 16;
    readInfo->bufOffset = 0;
    readInfo->resv = 0;

    __asc_simt_vf::RingBufWriteInfo* writeInfo = reinterpret_cast<__asc_simt_vf::RingBufWriteInfo*>(g_sysSimtPrintFifoSpace + bufSize - sizeof(__asc_simt_vf::RingBufWriteInfo));
    writeInfo->type = static_cast<uint32_t>(__asc_simt_vf::DumpType::DUMP_BUFI);
    writeInfo->length = 16;
    writeInfo->bufOffset = 0;
    writeInfo->packIdx = 0;

    AscendC::Simt::PRINTF("bufSize: %u, %s", bufSize, "test");
    AscendC::Simt::printf("bufSize: %u, %s", bufSize, "test");
    __asc_simt_vf::printf("bufSize: %u, %s", bufSize, "test");

    PrintTlvCheck* tlv1 = reinterpret_cast<PrintTlvCheck*>(blockInfo->ringBufAddr);
    PrintTlvCheck* tlv2 = tlv1 + 1;
    EXPECT_EQ(tlv1->type, tlv2->type);
    EXPECT_EQ(tlv1->length, tlv2->length);
    EXPECT_EQ(tlv1->blockIdx[0], tlv2->blockIdx[0]);
    EXPECT_EQ(tlv1->blockIdx[1], tlv2->blockIdx[1]);
    EXPECT_EQ(tlv1->blockIdx[2], tlv2->blockIdx[2]);
    EXPECT_EQ(tlv1->threadIdx[0], tlv2->threadIdx[0]);
    EXPECT_EQ(tlv1->threadIdx[1], tlv2->threadIdx[1]);
    EXPECT_EQ(tlv1->threadIdx[2], tlv2->threadIdx[2]);
    EXPECT_EQ(tlv1->fmtOffset, tlv2->fmtOffset);
    EXPECT_EQ(tlv1->param[0], tlv2->param[0]);
    EXPECT_EQ(tlv1->param[1], tlv2->param[1]);

    for (int i = 0; i < 100; i++) {
        __asc_simt_vf::printf("bufSize: %u, %s", bufSize, "test");
        readInfo->bufOffset += sizeof(PrintTlvCheck);
    }
    __asc_simt_vf::RingBufReadInfo readInfo1 = {0, 0, 100, 0};
    __asc_simt_vf::ring_buffer_wait(&readInfo1, 0);
    __asc_simt_vf::write_finish(blockInfo, blockInfo->ringBufLen - 2, __asc_simt_vf::DumpType::DUMP_SIMT_PRINTF);
}
