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

namespace AscendC {
class SubBlockNumAICTest : public testing::Test {
protected:
    void SetUp() {
        AscendC::SetGCoreType(1);
    }
    void TearDown() {
        AscendC::CheckSyncState();
        AscendC::SetGCoreType(0);
    }
};
TEST_F(SubBlockNumAICTest, TestSubBlockNum)
{
    EXPECT_EQ(GetSubBlockNum(), 1);
    EXPECT_EQ(GetTaskRation(), 1);
}

class SubBlockNumAIVTest : public testing::Test {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::CheckSyncState();
        AscendC::SetGCoreType(0);
    }
};
TEST_F(SubBlockNumAIVTest, TestSubBlockNum)
{
    EXPECT_EQ(GetSubBlockNum(), 2);
    EXPECT_EQ(GetTaskRation(), 2);
}
} // namespace AscendC