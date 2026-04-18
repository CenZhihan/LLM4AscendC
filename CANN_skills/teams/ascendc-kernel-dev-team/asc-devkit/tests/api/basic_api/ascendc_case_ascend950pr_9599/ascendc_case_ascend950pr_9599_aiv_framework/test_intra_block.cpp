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
#include "stub_def.h"
#include "stub_fun.h"
#include "securec.h"
using namespace std;

class TestIntraBlock: public testing::Test {
protected:
    void SetUp() {
        block_idx = 0;
        block_num = 1;
        g_kernelMode = KernelMode::MIX_MODE;
        set_core_type(0);
        g_coreType = 1;
        g_taskRation = 2;
        sub_block_idx = 0;
        
    }
    void TearDown()
    {
    }
};

TEST_F(TestIntraBlock, MIXIntraBlock)
{
    constexpr size_t fftsCounterSize = AscendC::GetMaxCoreNum() *
        AscendC::MIX_IN_GROUP_CORE_NUM * AscendC::FLAG_NUM * AscendC::FFTS_COUNTER_NUM;
    void* ffts_addr = AscendC::GmAlloc(fftsCounterSize);
    memset_s(ffts_addr, fftsCounterSize, 0, fftsCounterSize);
    set_ffts_base_addr(reinterpret_cast<uint64_t>(ffts_addr));
    AscendC::CheckNumBlocksForFfts(1);
    set_intra_block(static_cast<pipe_t>(0), 0);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[0][0], 1);
    set_intra_block(static_cast<pipe_t>(0), 16);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[1][0], 1);
    g_coreType = 2;
    wait_intra_block(static_cast<pipe_t>(0), 0);
    set_intra_block(static_cast<pipe_t>(0), 0);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[0][0], 0);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[72][0], 1);
    sub_block_idx = 1;
    wait_intra_block(static_cast<pipe_t>(0), 0);
    set_intra_block(static_cast<pipe_t>(0), 0);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[1][0], 0);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[72][16], 1);
    g_coreType = 1;
    wait_intra_block(static_cast<pipe_t>(0), 0);
    wait_intra_block(static_cast<pipe_t>(0), 16);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[72][0], 0);
    EXPECT_EQ(AscendC::g_syncCounterEachcore[72][16], 0);
}
