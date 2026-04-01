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
#include <sys/mman.h>
#define private public
#define protected public
#include "kernel_operator.h"
#include "mockcpp/mockcpp.hpp"

using namespace std;
using namespace AscendC;

class TestProcessLock : public testing::Test {
protected:
    void SetUp()
    {
        ProcessLock::processLock = nullptr;
    }
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TestProcessLock, ProcessLockConstruct)
{
    ProcessLock::CreateLock();
    EXPECT_TRUE(ProcessLock::processLock != nullptr);
}

TEST_F(TestProcessLock, ProcessLockFree)
{
    ProcessLock::CreateLock();
    EXPECT_TRUE(ProcessLock::processLock != nullptr);
    ProcessLock::FreeLock();
    EXPECT_TRUE(ProcessLock::processLock == nullptr);
    ProcessLock::FreeLock();
    EXPECT_TRUE(ProcessLock::processLock == nullptr);
    ProcessLock::GetProcessLock();
    EXPECT_TRUE(ProcessLock::processLock != nullptr);
    ProcessLock::GetProcessLock();
    EXPECT_TRUE(ProcessLock::processLock != nullptr);
    ProcessLock::FreeLock();
    EXPECT_TRUE(ProcessLock::processLock == nullptr);
    ProcessLock::FreeLock();
    EXPECT_TRUE(ProcessLock::processLock == nullptr);
}

void* MmapStub0(void* start,size_t length,int prot,int flags,int fd,off_t offset)
{
    return ((void*)-1);
}
int32_t MunmapStub0(void* start,size_t length)
{
    return -1;
}
int32_t RaiseStub0(int32_t i)
{
    return 0;
}
TEST_F(TestProcessLock, ProcessLockUnmapFail)
{
    MOCKER(munmap, int (*)(void*, size_t)).times(1).will(invoke(MunmapStub0));
    ProcessLock::GetProcessLock(); // 需要先申请printLock内存，不然走不到munmap
    EXPECT_TRUE(ProcessLock::processLock != nullptr);
    ProcessLock::FreeLock();
}
TEST_F(TestProcessLock, ProcessLockMapFail)
{
    MOCKER(mmap, void* (*)(void*, size_t, int, int, int, off_t)).times(1).will(invoke(MmapStub0));
    MOCKER(raise, int32_t (*)(int32_t)).times(1).will(invoke(RaiseStub0));
    ProcessLock::GetProcessLock();
    EXPECT_TRUE(static_cast<int>(reinterpret_cast<intptr_t>(ProcessLock::processLock)) == -1);
    ProcessLock::processLock = nullptr;
}