/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>  
#include <iostream>                
#include "kernel_operator.h"      
#include "kernel_utils.h"
#include "common.h"
#include "mockcpp/mockcpp.hpp"

namespace GetSsbufBaseAddrTest {
class TestGetSsbufBaseAddr : public testing::Test {
protected:
    void SetUp() {
        g_kernelMode = KernelMode::MIX_MODE;
    }
    void TearDown() {
    }
};

TEST_F(TestGetSsbufBaseAddr, GetSsbufBaseAddrMIXMode)
{
    void* ssbufAddr = AscendC::GetSsbufBaseAddr();
    EXPECT_EQ(ssbufAddr, reinterpret_cast<void*>(AscendC::ConstDefiner::Instance().cpuSSbuf));
}
}