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
#include "mockcpp/mockcpp.hpp"
#include "kernel_operator.h"
#include "securec.h"

using namespace std;
using namespace AscendC;

struct TestKfcClientParams {
    uint64_t blockIdx;
    uint32_t sendHeadOffset;
};

class TestKfcClientSuite : public testing::Test, public testing::WithParamInterface<TestKfcClientParams> {
public:
uint8_t* workspace;
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
        const int size = 16*1024*1024;
        workspace = new uint8_t[size];
        memset_s(workspace, size, static_cast<uint8_t>(AscendC::MSG_STATE::STATE_INVALID), size);
    }
    void TearDown() {
        AscendC::CheckSyncState();
        delete[] workspace;
        AscendC::SetGCoreType(1);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_KFC_CLIENT, TestKfcClientSuite,
    ::testing::Values(
    TestKfcClientParams { 0, 0 * MAX_MSG_COUNT * sizeof(AscendC::KfcMsg) }
));

TEST_P(TestKfcClientSuite, TestKfcClientConstruct)
{                
    AscendC::TPipe tpipe;           
    uint32_t flag = 0;
    uint32_t KfcMsgState = AscendC::KfcMsgGetState(flag);
    AscendC::KFC_Enum funID = AscendC::KFC_Enum::MMFUN_INIT;
    uint16_t instID = 0x1234;
    uint32_t KfcMsgFlag =  AscendC::KfcMsgMakeFlag(funID, instID);
    auto ret = AscendC::GetKfcClient(); // __DAV_C220_CUBE__
    EXPECT_EQ(ret, nullptr);
}
