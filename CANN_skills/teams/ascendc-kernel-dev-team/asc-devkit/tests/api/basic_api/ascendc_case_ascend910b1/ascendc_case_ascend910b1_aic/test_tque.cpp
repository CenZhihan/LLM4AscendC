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

using namespace std;
using namespace AscendC;

class TEST_TSCM: public testing::Test {
protected:
    void SetUp() {
        g_coreType = AscendC::AIC_TYPE;
    }
    void TearDown() {
        AscendC::CheckSyncState();
        g_coreType = AscendC::MIX_TYPE;
    }
};

TEST_F(TEST_TSCM, TEST_ALLOC_AND_FREE_BUFFER)
{
    static constexpr TPosition tpTscm[8] = {TPosition::MAX,
        TPosition::MAX,
        TPosition::MAX,
        TPosition::MAX,
        TPosition::MAX,
        TPosition::MAX,
        TPosition::MAX,
        TPosition::MAX};
    static constexpr TQueConfig confTscm = GetTQueConfig(false, false, false, 0, 0, 0, tpTscm, false, true);

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
}