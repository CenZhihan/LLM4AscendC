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
#ifdef ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_check/utils/init_global_memory/init_global_memory_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class InitGlobalMemoryAPICheck : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        AscendC::KernelRaise::GetInstance().SetRaiseMode(false);
    }
    void TearDown() {
        AscendC::KernelRaise::GetInstance().SetRaiseMode(true);
    }
};

TEST_F(InitGlobalMemoryAPICheck, InitGlobalMemoryAPICheckTestSuccess)
{
    __gm__ uint8_t* __restrict__ srcGm;
    AscendC::GlobalTensor<float> dstTensor;
    dstTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(srcGm), 32);

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(InitGlobalMemory, (float), (dstTensor, 8, 1));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(InitGlobalMemoryAPICheck, InitGlobalMemoryAPICheckTestFailure)
{
    __gm__ uint8_t* __restrict__ srcGm;
    AscendC::GlobalTensor<uint8_t> dstTensor;
    dstTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(srcGm), 32);

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(InitGlobalMemory, (uint8_t), (dstTensor, 32, 1));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}
