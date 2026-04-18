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
using namespace AscendC;

class TEST_SYS_VAR : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TEST_SYS_VAR, GetSystemVariable) {
    const int64_t virtualAddress = 0;

    uint32_t coreVersion = 0;
    GetArchVersion(coreVersion);
    int64_t pc = -1;
    pc = GetProgramCounter();
    EXPECT_NE(pc, -1);
}