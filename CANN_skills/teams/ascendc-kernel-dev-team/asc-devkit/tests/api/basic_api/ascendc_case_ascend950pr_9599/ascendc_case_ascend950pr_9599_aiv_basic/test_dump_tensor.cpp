/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#define ASCENDC_DUMP 1
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <vector>
#include "kernel_operator.h"
// #include "api_check/kernel_cpu_check.h"

using namespace AscendC;

int32_t RaiseStubForDumpTensor(int32_t input);

struct TestDumpTensorParams {
    uint32_t dataSize;
    uint32_t typeSize;
    uint32_t dstSize;
    uint64_t dumpSize;
    void (*CalFunc)(uint8_t *, uint8_t *, uint8_t *, uint32_t, uint32_t, uint64_t);
};

class TestDumpTensorSuite : public testing::Test, public testing::WithParamInterface<TestDumpTensorParams> {
protected:
    void SetUp()
    {
        MOCKER(raise, int32_t (*)(int32_t)).stubs().will(invoke(RaiseStubForDumpTensor));
    }
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

class DumpTensorRuntimeGuard {
public:
    DumpTensorRuntimeGuard() :
        coreType_(g_coreType), blockNum_(block_num), blockIdx_(block_idx), subBlockIdx_(sub_block_idx),
        taskRation_(g_taskRation), dumpWorkspace_(AscendC::g_dumpWorkspaceReserved), sysWorkspace_(GetSysWorkSpacePtr())
    {}

    ~DumpTensorRuntimeGuard()
    {
        SetGCoreType(coreType_);
        block_num = blockNum_;
        block_idx = blockIdx_;
        sub_block_idx = subBlockIdx_;
        g_taskRation = taskRation_;
        AscendC::g_dumpWorkspaceReserved = dumpWorkspace_;
        SetSysWorkSpacePtr(sysWorkspace_);
    }

private:
    int32_t coreType_;
    decltype(block_num) blockNum_;
    decltype(block_idx) blockIdx_;
    decltype(sub_block_idx) subBlockIdx_;
    decltype(g_taskRation) taskRation_;
    __gm__ uint8_t* dumpWorkspace_;
    __gm__ uint8_t* sysWorkspace_;
};

void SetAivDumpEnv(uint32_t blockNumValue, uint32_t blockIdxValue, uint32_t subBlockIdxValue, uint32_t taskRationValue)
{
    SetGCoreType(2);
    block_num = blockNumValue;
    block_idx = blockIdxValue;
    sub_block_idx = subBlockIdxValue;
    g_taskRation = taskRationValue;
}

template <typename T>
void DataDumpTensor(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, __gm__ uint8_t *workGm, __gm__ uint32_t dataSize,
    __gm__ uint32_t dstSize, __gm__ uint64_t dumpSize)
{
    TPipe tpipe;
    uint32_t desc = 630;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    GlobalTensor<T> workGlobal;

    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), dstSize);
    workGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workGm), DUMP_WORKSPACE_SIZE * 2);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();
    DataCopyParams copyParams{1, dataSize * sizeof(T) / 32, 0, 0};
    DataCopy(inputLocal, srcGlobal, copyParams);
    pipe_barrier(PIPE_ALL);

    InitDump(false, workGm, dumpSize);
}

INSTANTIATE_TEST_CASE_P(TEST_DUMP_TENSOR, TestDumpTensorSuite,
    ::testing::Values(TestDumpTensorParams{64, 4, 96, DUMP_UINTSIZE, DataDumpTensor<uint32_t>}));

int32_t RaiseStubForDumpTensor(int32_t input)
{
    return 0;
}

TEST_P(TestDumpTensorSuite, TestDumpTensorCases)
{
    auto param = GetParam();
    int32_t coreTypeTmp = g_coreType;
    auto blockNumTmp = block_num;
    auto blockIdxTmp = block_idx;
    auto subBlockIdxTmp = sub_block_idx;
    auto taskRationTmp = g_taskRation;
    SetGCoreType(2);
    block_num = 1;
    block_idx = 0;
    sub_block_idx = 0;
    g_taskRation = 1;

    uint8_t srcGm[param.dataSize * param.typeSize];
    uint8_t dstGm[param.dstSize * param.typeSize];
    std::vector<uint8_t> workGm(param.dumpSize, 0);

    param.CalFunc(dstGm, srcGm, workGm.data(), param.dataSize, param.dstSize, param.dumpSize);

    auto* blockInfo = reinterpret_cast<uint32_t*>(workGm.data());
    EXPECT_EQ(blockInfo[BLOCK_INFO_MAGIC_POS], BLOCK_INFO_MAGIC_NUM);

    SetGCoreType(coreTypeTmp);
    block_num = blockNumTmp;
    block_idx = blockIdxTmp;
    sub_block_idx = subBlockIdxTmp;
    g_taskRation = taskRationTmp;
}

TEST_F(TestDumpTensorSuite, InitDumpImplNullWorkspaceReturns)
{
    DumpTensorRuntimeGuard guard;
    SetAivDumpEnv(1, 0, 0, 1);

    AscendC::g_dumpWorkspaceReserved = nullptr;
    InitDumpImpl(false, DUMP_UINTSIZE);

    EXPECT_EQ(AscendC::g_dumpWorkspaceReserved, nullptr);
}

TEST_F(TestDumpTensorSuite, InitDumpImplMixFlagWritesMixedBlockNum)
{
    DumpTensorRuntimeGuard guard;
    SetAivDumpEnv(1, 0, 0, 1);

    std::vector<uint8_t> workGm(DUMP_UINTSIZE, 0);
    AscendC::g_dumpWorkspaceReserved = workGm.data();
    InitDumpImpl(true, DUMP_UINTSIZE);

    auto* blockInfo = reinterpret_cast<uint32_t*>(workGm.data());
    EXPECT_EQ(blockInfo[BLOCK_INFO_MAGIC_POS], BLOCK_INFO_MAGIC_NUM);
    EXPECT_EQ(blockInfo[BLOCK_INFO_BLOCKNUM_POS], 3U);
}

TEST_F(TestDumpTensorSuite, InitDumpImplSkipsOutOfRangeCore)
{
    DumpTensorRuntimeGuard guard;
    SetAivDumpEnv(1, 54, 0, 2);

    std::vector<uint8_t> workGm(DUMP_UINTSIZE, 0);
    AscendC::g_dumpWorkspaceReserved = workGm.data();
    EXPECT_EQ(GetDumpBlockIdx(), DUMP_CORE_COUNT);

    InitDumpImpl(false, DUMP_UINTSIZE);

    auto* blockInfo = reinterpret_cast<uint32_t*>(workGm.data());
    EXPECT_EQ(blockInfo[BLOCK_INFO_MAGIC_POS], 0U);
}

TEST_F(TestDumpTensorSuite, InitDumpNullStartAddrReturns)
{
    DumpTensorRuntimeGuard guard;
    SetAivDumpEnv(1, 0, 0, 1);

    InitDump(false, static_cast<__gm__ uint8_t*>(nullptr), DUMP_UINTSIZE);

    EXPECT_EQ(AscendC::g_dumpWorkspaceReserved, nullptr);
}
