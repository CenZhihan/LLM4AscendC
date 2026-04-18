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
#include "kernel_operator.h"
 
using namespace std;
using namespace AscendC;
 
class TestKernelReg : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {
        AscendC::CheckSyncState();
        GlobalMockObject::verify();
    }
};
 
TEST_F(TestKernelReg, TestPipeBarrier)
{   
    int32_t tmp = g_coreType;
    g_coreType = AscendC::AIC_TYPE;
    MOCKER(pipe_barrier).expects(never());
    PipeBarrierImpl<PIPE_V>();
    EXPECT_EQ(g_coreType, AscendC::AIC_TYPE);
    g_coreType = tmp;
}