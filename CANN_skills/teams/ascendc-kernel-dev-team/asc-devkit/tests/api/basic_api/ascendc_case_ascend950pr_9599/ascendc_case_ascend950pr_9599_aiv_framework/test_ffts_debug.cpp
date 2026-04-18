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
#include "kernel_operator.h"
#include "stub_def.h"
#include "stub_fun.h"
#include "securec.h"
#include <stdio.h>  
#include <unistd.h>  
#include <sys/types.h>  
#include <sys/wait.h>  

using namespace std;

class TestFfts: public testing::Test {
protected:
    void SetUp() {
        block_idx = 0;
        block_num = 1;
        g_coreType = 1;
        g_taskRation = 2;
        sub_block_idx = 0;
        g_kernelMode = KernelMode::MIX_MODE;
    }
    void TearDown()
    {
    }
};

TEST_F(TestFfts, MIXFftsCaseMode0)
{
    constexpr size_t fftsCounterSize = AscendC::GetMaxCoreNum() *
        AscendC::MIX_IN_GROUP_CORE_NUM * AscendC::FLAG_NUM * AscendC::FFTS_COUNTER_NUM;
    void* ffts_addr = AscendC::GmAlloc(fftsCounterSize);
    memset_s(ffts_addr, fftsCounterSize, 0, fftsCounterSize);
    set_ffts_base_addr(reinterpret_cast<uint64_t>(ffts_addr));
    AscendC::CheckNumBlocksForFfts(1);
    ffts_cross_core_sync(static_cast<pipe_t>(0), 0x301);
    EXPECT_EQ(AscendC::g_syncCounterFfts[72][3], 0);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[72][3], 1);
    wait_flag_dev(PIPE_MTE3, 0x3l);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[72][3], 0);
    g_coreType = 2;
    ffts_cross_core_sync(static_cast<pipe_t>(0), 0x301);
    sub_block_idx = 1;
    ffts_cross_core_sync(static_cast<pipe_t>(0), 0x301);
    EXPECT_EQ(AscendC::g_syncCounterFfts[0][3], 0);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[0][3], 1);
    sub_block_idx = 0;
    wait_flag_dev(PIPE_MTE3, 0x3l);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[0][3], 0);
    sub_block_idx = 1;
    wait_flag_dev(PIPE_MTE3, 0x3l);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[1][3], 0);
    AscendC::GmFree((void*)AscendC::g_syncCounterFfts);
}

TEST_F(TestFfts, AICFftsCaseMode0)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    g_coreType = 1;
    constexpr size_t fftsCounterSize = AscendC::GetMaxCoreNum() *
        AscendC::MIX_IN_GROUP_CORE_NUM * AscendC::FLAG_NUM * AscendC::FFTS_COUNTER_NUM;
    void* ffts_addr = AscendC::GmAlloc(fftsCounterSize);
    memset_s(ffts_addr, fftsCounterSize, 0, fftsCounterSize);
    set_ffts_base_addr(reinterpret_cast<uint64_t>(ffts_addr));
    AscendC::CheckNumBlocksForFfts(1);
    ffts_cross_core_sync(static_cast<pipe_t>(0), 0x301);
    EXPECT_EQ(AscendC::g_syncCounterFfts[72][3], 0);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[72][3], 1);
    wait_flag_dev(PIPE_MTE3, static_cast<uint16_t>(0x3));
    EXPECT_EQ(AscendC::g_syncCounterEachcore[72][3], 0);
    AscendC::GmFree((void*)AscendC::g_syncCounterFfts);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
}

TEST_F(TestFfts, AIVFftsCaseMode0)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    g_coreType = 2;
    constexpr size_t fftsCounterSize = AscendC::GetMaxCoreNum() *
        AscendC::MIX_IN_GROUP_CORE_NUM * AscendC::FLAG_NUM * AscendC::FFTS_COUNTER_NUM;
    void* ffts_addr = AscendC::GmAlloc(fftsCounterSize);
    memset_s(ffts_addr, fftsCounterSize, 0, fftsCounterSize);
    set_ffts_base_addr(reinterpret_cast<uint64_t>(ffts_addr));
    AscendC::CheckNumBlocksForFfts(1);
    ffts_cross_core_sync(static_cast<pipe_t>(0), 0x301);
    EXPECT_EQ(AscendC::g_syncCounterFfts[0][3], 0);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[0][3], 1);
    wait_flag_dev(PIPE_MTE3, 0x3l);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[0][3], 0);
    AscendC::GmFree((void*)AscendC::g_syncCounterFfts);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
}

TEST_F(TestFfts, FftsCaseMode1)
{
    constexpr size_t fftsCounterSize = AscendC::GetMaxCoreNum() *
        AscendC::MIX_IN_GROUP_CORE_NUM * AscendC::FLAG_NUM * AscendC::FFTS_COUNTER_NUM;
    void* ffts_addr = AscendC::GmAlloc(fftsCounterSize);
    memset_s(ffts_addr, fftsCounterSize, 0, fftsCounterSize);
    set_ffts_base_addr(reinterpret_cast<uint64_t>(ffts_addr));
    AscendC::CheckNumBlocksForFfts(1);
    g_coreType = 2;
    sub_block_idx = 0;
    ffts_cross_core_sync(static_cast<pipe_t>(0), 0x311);
    EXPECT_EQ(AscendC::g_syncCounterFfts[0][3], 1);
    sub_block_idx = 1;
    ffts_cross_core_sync(static_cast<pipe_t>(0), 0x311);
    EXPECT_EQ(AscendC::g_syncCounterFfts[0][3], 0);
    EXPECT_EQ(AscendC::g_syncCounterFfts[1][3], 0);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[0][3], 1);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[1][3], 1);
    sub_block_idx = 0;
    wait_flag_dev(PIPE_MTE3, 0x3l);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[0][3], 0);
    sub_block_idx = 1;
    wait_flag_dev(PIPE_MTE3, 0x3l);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[1][3], 0);
    AscendC::GmFree((void*)AscendC::g_syncCounterFfts);
}

TEST_F(TestFfts, FftsCaseMode2)
{
    constexpr size_t fftsCounterSize = AscendC::GetMaxCoreNum() *
        AscendC::MIX_IN_GROUP_CORE_NUM * AscendC::FLAG_NUM * AscendC::FFTS_COUNTER_NUM;
    void* ffts_addr = AscendC::GmAlloc(fftsCounterSize);
    memset_s(ffts_addr, fftsCounterSize, 0, fftsCounterSize);
    set_ffts_base_addr(reinterpret_cast<uint64_t>(ffts_addr));

    block_idx = 0;
    block_num = 1;
    g_coreType = 1;
    g_taskRation = 2;
    sub_block_idx = 0;
    g_kernelMode = KernelMode::MIX_MODE;

    AscendC::CheckNumBlocksForFfts(1);
    ffts_cross_core_sync(static_cast<pipe_t>(0), 0x321);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[0][3], 1);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[1][3], 1);
    g_coreType = 2;
    sub_block_idx = 0;
    wait_flag_dev(PIPE_MTE3, 0x3l);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[0][3], 0);
    sub_block_idx = 1;
    wait_flag_dev(PIPE_MTE3, 0x3l);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[1][3], 0);
    sub_block_idx = 0;
    ffts_cross_core_sync(static_cast<pipe_t>(0), 0x321);
    EXPECT_EQ(AscendC::g_syncCounterFfts[0][3], 1);
    EXPECT_EQ(AscendC::g_syncCounterFfts[72][3], 0);
    sub_block_idx = 1;
    ffts_cross_core_sync(static_cast<pipe_t>(0), 0x321);
    EXPECT_EQ(AscendC::g_syncCounterFfts[0][3], 0);
    EXPECT_EQ(AscendC::g_syncCounterFfts[1][3], 0);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[72][3], 1);
    sub_block_idx = 0;
    g_coreType = 1;
    wait_flag_dev(PIPE_MTE3, 0x3l);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[72][3], 0);
    AscendC::GmFree((void*)AscendC::g_syncCounterFfts);
}

TEST_F(TestFfts, MultiProcessTest)
{
    constexpr size_t fftsCounterSize = AscendC::GetMaxCoreNum() *
        AscendC::MIX_IN_GROUP_CORE_NUM * AscendC::FLAG_NUM * AscendC::FFTS_COUNTER_NUM;
    void* ffts_addr = AscendC::GmAlloc(fftsCounterSize);
    memset_s(ffts_addr, fftsCounterSize, 0, fftsCounterSize);
    set_ffts_base_addr(reinterpret_cast<uint64_t>(ffts_addr));

    block_idx = 0;
    block_num = 1;
    g_coreType = 1;
    g_taskRation = 2;
    sub_block_idx = 0;
    g_kernelMode = KernelMode::MIX_MODE;

    pid_t pid1 = fork();
    if (pid1 < 0) {
        perror("fork error!");
    } else if (pid1 == 0) {
        g_coreType = 2;
        sub_block_idx = 0;
        wait_flag_dev(PIPE_MTE3, 0x3l);
        exit(0);
    }
    
    pid_t pid2 = fork();
    if (pid2 < 0) {
        perror("fork error!");
    } else if (pid2 == 0) {
        g_coreType = 1;
        sub_block_idx = 0;
        ffts_cross_core_sync(static_cast<pipe_t>(0), 0x321);
        exit(0);
    }

    // 父进程  
    printf("This is the parent process\n");  
  
    // 等待子进程结束  
    wait(nullptr);  
    wait(nullptr);  
  
    printf("All child processes have exited\n");  
    AscendC::GmFree((void*)AscendC::g_syncCounterFfts);
}