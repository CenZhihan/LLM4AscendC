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

struct TestKfcServerParams {
    uint64_t blockIdx;
    uint32_t sendHeadOffset;
};

class TestKfcServerSuite : public testing::Test, public testing::WithParamInterface<TestKfcServerParams> {
public:
uint8_t* workspace;
TPipe tpipe;
protected:
    void SetUp() {
        const int size = 16 * 1024 * 1024;
        workspace = new uint8_t[size];
        memset_s(workspace, size, static_cast<uint8_t>(MSG_STATE::STATE_INVALID), size);
    }
    void TearDown() {
        delete[] workspace;
    }
};

INSTANTIATE_TEST_CASE_P(TestKfcServer, TestKfcServerSuite,
    ::testing::Values(
    TestKfcServerParams { 0, 0 * MAX_MSG_COUNT * sizeof(KfcMsg) }
));

TEST_P(TestKfcServerSuite, TestKfcServerConstruct)
{
    int32_t gCoreTypeTemp = g_coreType;
    AscendC::SetGCoreType(2);
    auto param = GetParam();
    constexpr int32_t MSG_OFFSET = MAX_MSG_COUNT * sizeof(KfcMsg);
    MOCKER_CPP(GetBlockIdxImpl).stubs().will(returnValue(param.blockIdx));
    KfcCommServer kfcAIC;
    kfcAIC.Init(workspace, param.blockIdx);
    EXPECT_EQ(reinterpret_cast<uint8_t*>(kfcAIC.msgSendStart), workspace + (param.blockIdx * 2 + 1) * MSG_OFFSET);
    EXPECT_EQ(reinterpret_cast<uint8_t*>(kfcAIC.msgRcvStart), workspace + param.blockIdx * MSG_OFFSET * 2);
    EXPECT_EQ(reinterpret_cast<uint8_t*>(kfcAIC.msgSendHead), workspace + (param.blockIdx * 2 + 1) * MSG_OFFSET);
    EXPECT_EQ(reinterpret_cast<uint8_t*>(kfcAIC.msgRcvHead), workspace + param.blockIdx * MSG_OFFSET * 2);
    EXPECT_EQ(kfcAIC.msgSendPos, 0);
    EXPECT_EQ(kfcAIC.msgRcvPos, 0);
    __gm__ KfcMsg * msg = kfcAIC.AllocMessage();
    msg->head = 1;
    AscendC::SetGCoreType(1);
    auto msgHead = GetMsgHead(workspace, 0);
    AscendC::SetGCoreType(gCoreTypeTemp);

    uint32_t flag = 0;
    uint32_t KfcMsgState = KfcMsgGetState(flag);
    KFC_Enum funID = KFC_Enum::MMFUN_INIT;
    uint16_t instID = 0x1234;
    uint32_t KfcMsgFlag =  KfcMsgMakeFlag(funID, instID);
}