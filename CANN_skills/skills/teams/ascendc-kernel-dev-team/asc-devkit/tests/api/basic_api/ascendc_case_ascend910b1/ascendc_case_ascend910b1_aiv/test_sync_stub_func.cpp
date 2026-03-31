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

#include "kernel_operator.h"

using namespace std;
// using namespace AscendC;
namespace AscendC {
void __ib_set_stub(int32_t blockIdx, int32_t eventID, bool isAIVOnly);
void __ib_wait_stub(int32_t blockIdx, int32_t eventID, bool isAIVOnly);
void __sync_all_stub(int32_t usedCores, bool isAIVOnly);
}

class TestSyncStubFunc : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {}
};


TEST_F(TestSyncStubFunc, SyncTestCase)
{
    int32_t blockIdx = 0;
    int32_t eventId = 0;
    bool isAIVOnly = true;
    int32_t usedCores = 1;
    AscendC::__ib_set_stub(blockIdx, eventId, isAIVOnly);
    AscendC::__ib_wait_stub(blockIdx, eventId, isAIVOnly);
    AscendC::__sync_all_stub(usedCores, isAIVOnly);
    EXPECT_TRUE(isAIVOnly);
}
