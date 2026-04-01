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
#include "kernel_utils.h"
// #include "model/model_factory.h"
#include "common.h"
#include "mockcpp/mockcpp.hpp"

namespace AscendC {
template <pipe_t triggerPipe, pipe_t waitPipe, bool defaultConfig>
__global__ __aicore__ void HardSyncAllTest(pipe_t tPipe, pipe_t wPipe, pipe_t bPipe)
{
    static constexpr SyncAllConfig config = {triggerPipe, waitPipe};
    pipe_t spyTriggerPipe;
    pipe_t spyWaitPipe;
    pipe_t spyBarrierPipe;
    MOCKER_CPP(ffts_cross_core_sync, void(*)(pipe_t, uint64_t))
        .stubs()
        .with(spy(spyTriggerPipe), any());
    MOCKER_CPP(wait_flag_dev, void(*)(pipe_t, uint16_t))
        .stubs()
        .with(spy(spyWaitPipe), any());
    MOCKER_CPP(pipe_barrier, void(*)(pipe_t))
        .stubs()
        .with(spy(spyBarrierPipe));

    const pipe_t expectedTriggerPipe = tPipe;
    const pipe_t expectedWaitPipe = wPipe;
    const pipe_t expectedBarrierPipe = bPipe;
    if constexpr (defaultConfig) {
        SyncAll();
    } else {
        SyncAll<true, config>();
    }

    EXPECT_EQ(spyTriggerPipe, expectedTriggerPipe);
    EXPECT_EQ(spyWaitPipe, expectedWaitPipe);
    EXPECT_EQ(spyBarrierPipe, expectedBarrierPipe);
}

struct SyncAllTestParams {
    pipe_t fftsPipe;
    pipe_t waitPipe;
    pipe_t barrierPipe;
    void (*cal_func)(pipe_t, pipe_t, pipe_t);
};

class SyncAllTestsuite : public testing::Test, public testing::WithParamInterface<SyncAllTestParams> {
protected:
    void SetUp() {
        SetGCoreType(2);
    }
    void TearDown() {
        GlobalMockObject::verify();
    }
};

INSTANTIATE_TEST_CASE_P(TestSyncAll, SyncAllTestsuite,
    ::testing::Values(
        SyncAllTestParams { PIPE_MTE2, PIPE_MTE2, PIPE_MTE2, HardSyncAllTest<PIPE_MTE2, PIPE_MTE2, false> },
        SyncAllTestParams { PIPE_MTE2, PIPE_MTE3, PIPE_MTE2, HardSyncAllTest<PIPE_MTE2, PIPE_MTE3, false> },
        SyncAllTestParams { PIPE_MTE2, PIPE_S, PIPE_MTE2, HardSyncAllTest<PIPE_MTE2, PIPE_ALL, false> },
        SyncAllTestParams { PIPE_MTE3, PIPE_MTE2, PIPE_MTE3, HardSyncAllTest<PIPE_MTE3, PIPE_MTE2, false> },
        SyncAllTestParams { PIPE_MTE3, PIPE_MTE3, PIPE_MTE3, HardSyncAllTest<PIPE_MTE3, PIPE_MTE3, false> },
        SyncAllTestParams { PIPE_MTE3, PIPE_S, PIPE_MTE3, HardSyncAllTest<PIPE_MTE3, PIPE_ALL, false> },
        SyncAllTestParams { PIPE_MTE3, PIPE_MTE2, PIPE_ALL, HardSyncAllTest<PIPE_ALL, PIPE_MTE2, false> },
        SyncAllTestParams { PIPE_MTE3, PIPE_MTE3, PIPE_ALL, HardSyncAllTest<PIPE_ALL, PIPE_MTE3, false> },
        SyncAllTestParams { PIPE_MTE3, PIPE_S, PIPE_ALL, HardSyncAllTest<PIPE_ALL, PIPE_ALL, false> },
        SyncAllTestParams { PIPE_MTE3, PIPE_S, PIPE_ALL, HardSyncAllTest<PIPE_ALL, PIPE_ALL, true> }
        ));

TEST_P(SyncAllTestsuite, SyncAllOpTestCase)
{
    auto param = GetParam();
    param.cal_func(param.fftsPipe, param.waitPipe, param.barrierPipe);
}
} // namespace AscendC
