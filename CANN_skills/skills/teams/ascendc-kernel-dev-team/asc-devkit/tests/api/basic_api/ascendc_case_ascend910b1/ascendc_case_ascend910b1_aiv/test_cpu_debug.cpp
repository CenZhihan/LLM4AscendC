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
#include <mockcpp/mockcpp.hpp>
#include "stub_def.h"
#include "stub_fun.h"

#define ASCENDC_DUMP 0
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;

class TestCpuDebugApiCheck : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        block_idx = 0;
        block_num = 8;
        g_coreType = 0;
        g_taskRation = 2;
        sub_block_idx = 0;
        g_kernelMode = KernelMode::MIX_MODE;
        AscendC::CheckSyncState();
        GlobalMockObject::verify();
    }
};

void DumpTensorWrapper(const LocalTensor<float>& tensor, uint32_t desc, uint32_t dumpSize)
{
    AscendC::DumpTensor(tensor, 0, 1);
}

int32_t RaiseStubForDump(int32_t input)
{
    return 0;
}

TEST_F(TestCpuDebugApiCheck, CpuDebugDisableDumpCase)
{
    MOCKER(DumpTensorWrapper).expects(once());
    MOCKER(raise, int32_t (*)(int32_t)).stubs().will(invoke(RaiseStubForDump));
    int tmp = 0;

    printf("123, 456");
    PRINTF("tmp = %d.\n", tmp);

    LocalTensor<float> tensor;
    DumpTensor(tensor, 0, 1);
    DumpTensorWrapper(tensor, 0, 1);

    DumpAccChkPoint(tensor, 0, 0, 1);
    AscendC::DumpAccChkPoint(tensor, 0, 0, 1);

    assert(tmp == 0);
    assert(tmp != 0);
    assert(assert_flag);
    EXPECT_NO_THROW(GlobalMockObject::verify());
}

TEST_F(TestCpuDebugApiCheck, CpuDebugApiCheckCase)
{
    int idx = 4;
    set_core_type(idx);
    int coreType = g_coreType;
    set_block_dim(idx);
    int blockNum = get_block_num();
    int blockIdx = get_block_idx();
    int processNum = get_process_num();
    EXPECT_EQ(sub_block_idx, 0);
    EXPECT_EQ(g_taskRation, 2);
    EXPECT_EQ(blockNum, 8);
    EXPECT_EQ(blockIdx, 1);
    EXPECT_EQ(processNum, 24);
    idx = 3;
    set_core_type(idx);
    coreType = g_coreType;
    EXPECT_EQ(coreType, AscendC::AIC_TYPE);
    idx = 5;
    set_core_type(idx);
    EXPECT_EQ(sub_block_idx, 1);
    coreType = g_coreType;
    EXPECT_EQ(coreType, AscendC::AIV_TYPE);
    idx = 6;
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    set_block_dim(idx);
    set_core_type(idx);
    blockNum = get_block_num();
    blockIdx = get_block_idx();
    processNum = get_process_num();
    EXPECT_EQ(blockNum, 8);
    EXPECT_EQ(blockIdx, 6);
    EXPECT_EQ(processNum, 8);
    EXPECT_EQ(sub_block_idx, 0);
    EXPECT_EQ(g_taskRation, 1);
    EXPECT_EQ(g_coreType, AscendC::AIC_TYPE);
    idx = 7;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    set_block_dim(idx);
    set_core_type(idx);
    blockNum = get_block_num();
    blockIdx = get_block_idx();
    processNum = get_process_num();
    EXPECT_EQ(blockNum, 8);
    EXPECT_EQ(blockIdx, 7);
    EXPECT_EQ(processNum, 8);
    EXPECT_EQ(sub_block_idx, 0);
    EXPECT_EQ(g_taskRation, 1);
    EXPECT_EQ(g_coreType, AscendC::AIV_TYPE);
}