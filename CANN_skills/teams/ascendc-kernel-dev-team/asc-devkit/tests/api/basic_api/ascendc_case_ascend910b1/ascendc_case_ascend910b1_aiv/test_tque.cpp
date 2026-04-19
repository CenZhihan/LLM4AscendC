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
#include <fstream>
#include <iostream>
#include "kernel_operator.h"
#include "mockcpp/mockcpp.hpp"

using namespace std;
using namespace AscendC;

class TEST_TSCM: public testing::Test {
protected:
    void SetUp() {
        AscendC::SetGCoreType(AIV_TYPE);
    }
    void TearDown() {
        AscendC::CheckSyncState();
        AscendC::SetGCoreType(0);
        GlobalMockObject::verify();
    }
};


void AbortStub1()
{

}
TEST_F(TEST_TSCM, TEST_ALLOC_AND_FREE_BUFFER) {
    static constexpr TPosition tpTscm[8] = {TPosition::MAX, TPosition::MAX, TPosition::MAX, TPosition::MAX,
            TPosition::MAX, TPosition::MAX, TPosition::MAX, TPosition::MAX};
    static constexpr TQueConfig confTscm = GetTQueConfig(false, false, false, 0, 0, 0, tpTscm, false, true);
    int32_t tmpCore = g_coreType;
    g_coreType = AscendC::AIV_TYPE;
    TPipe pipe;
    TSCM<TPosition::VECIN, 1, &confTscm> scmUb;
    TSCM<TPosition::GM, 1, &confTscm> scmGm;
    pipe.InitBuffer(scmUb, 1, 1024);
    pipe.InitBuffer(scmGm, 1, 1024);
    auto tmp = scmUb.AllocTensor<float>();
    auto tmp1 = scmGm.AllocTensor<float>();
    EXPECT_EQ(scmUb.bufUsedCount, 1);
    EXPECT_EQ(scmGm.bufUsedCount, 1);
    scmUb.FreeTensor(tmp);
    scmGm.FreeTensor(tmp1);
    g_coreType = tmpCore;
}

TEST_F(TEST_TSCM, TEST_ALLOC_TOO_MUCH_BUFFER_IN_ONE_INITBUFFER) {
    g_coreType = AscendC::AIV_TYPE;
    TPipe pipe;
    TQue<TPosition::VECIN, 1> tque1;
    MOCKER(raise, int(*)(int)).times(2).will(returnValue(0));

    static int32_t count =0;
    std::string fileName = "print_ut_aiv_init_buffer" + std::to_string(getpid()) + "_" + std::to_string(count)+ ".txt";
    freopen(fileName.c_str(), "w", stdout);

    pipe.InitBuffer(tque1, 65, 1024);

    // 恢复printf
    fclose(stdout);
    freopen("/dev/tty", "w", stdout);
    freopen("/dev/tty", "r", stdin);

    // 校验真值
    std::ifstream resultFile(fileName, std::ios::in);
    std::stringstream streambuffer;
    streambuffer << resultFile.rdbuf();
    std::string resultString(streambuffer.str());
    std::string goldenStr = "Failed to check num value in InitBuffer, its valid range is 1 ~ 64, current value is 65.";
    resultFile.close();
    std::cout << "resultString is " << resultString  << std::endl;
    std::cout << "goldenStr is " << goldenStr  << std::endl;
    EXPECT_TRUE(resultString.find(goldenStr) != std::string::npos);
    EXPECT_EQ(remove(fileName.c_str()), 0);
}

// 预计分配到65会挂
TEST_F(TEST_TSCM, TEST_ALLOC_MANY_BUFFER_IN_MULTIPLE_INITBUFFER) {
    g_coreType = AscendC::AIV_TYPE;
    TPipe pipe;
    TQue<TPosition::VECIN, 1> tque1;
    TQue<TPosition::VECIN, 1> tque2;
    pipe.InitBuffer(tque1, 32, 1024);
    pipe.InitBuffer(tque2, 32, 1024);
}