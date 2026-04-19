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

class TEST_HSYNC : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(1);
    }
    void TearDown()
    {
        AscendC::CheckSyncState();
        AscendC::SetGCoreType(0);
    }
};

void HsyncWrapper()
{
    HSetFlag<HardEvent::MTE1_M, MemoryT::L0A, true>(EVENT_ID0);
    HWaitFlag<HardEvent::MTE1_M, MemoryT::L0A, true>(EVENT_ID0);
    HSetFlag<HardEvent::MTE1_M, MemoryT::L0A, false>(EVENT_ID0);
    HWaitFlag<HardEvent::MTE1_M, MemoryT::L0A, false>(EVENT_ID0);
    HSetFlag<HardEvent::MTE1_M, MemoryT::L0B, true>(EVENT_ID0);
    HWaitFlag<HardEvent::MTE1_M, MemoryT::L0B, true>(EVENT_ID0);
    HSetFlag<HardEvent::MTE1_M, MemoryT::L0B, false>(EVENT_ID0);
    HWaitFlag<HardEvent::MTE1_M, MemoryT::L0B, false>(EVENT_ID0);
    HSetFlag<HardEvent::MTE1_M, MemoryT::BIAS, true>(EVENT_ID0);
    HWaitFlag<HardEvent::MTE1_M, MemoryT::BIAS, true>(EVENT_ID0);
    HSetFlag<HardEvent::MTE1_M, MemoryT::BIAS, false>(EVENT_ID0);
    HWaitFlag<HardEvent::MTE1_M, MemoryT::BIAS, false>(EVENT_ID0);

    HSetFlag<HardEvent::M_MTE1, MemoryT::L0A, true>(EVENT_ID0);
    HWaitFlag<HardEvent::M_MTE1, MemoryT::L0A, true>(EVENT_ID0);
    HSetFlag<HardEvent::M_MTE1, MemoryT::L0A, false>(EVENT_ID0);
    HWaitFlag<HardEvent::M_MTE1, MemoryT::L0A, false>(EVENT_ID0);
    HSetFlag<HardEvent::M_MTE1, MemoryT::L0B, true>(EVENT_ID0);
    HWaitFlag<HardEvent::M_MTE1, MemoryT::L0B, true>(EVENT_ID0);
    HSetFlag<HardEvent::M_MTE1, MemoryT::L0B, false>(EVENT_ID0);
    HWaitFlag<HardEvent::M_MTE1, MemoryT::L0B, false>(EVENT_ID0);
    HSetFlag<HardEvent::M_MTE1, MemoryT::BIAS, true>(EVENT_ID0);
    HWaitFlag<HardEvent::M_MTE1, MemoryT::BIAS, true>(EVENT_ID0);
    HSetFlag<HardEvent::M_MTE1, MemoryT::BIAS, false>(EVENT_ID0);
    HWaitFlag<HardEvent::M_MTE1, MemoryT::BIAS, false>(EVENT_ID0);

    HSetFlag<HardEvent::M_FIX, MemoryT::L0C, true>(EVENT_ID0);
    HWaitFlag<HardEvent::M_FIX, MemoryT::L0C, true>(EVENT_ID0);
    HSetFlag<HardEvent::M_FIX, MemoryT::L0C, false>(EVENT_ID0);
    HWaitFlag<HardEvent::M_FIX, MemoryT::L0C, false>(EVENT_ID0);

    HSetFlag<HardEvent::FIX_M, MemoryT::L0C, true>(EVENT_ID0);
    HWaitFlag<HardEvent::FIX_M, MemoryT::L0C, true>(EVENT_ID0);
    HSetFlag<HardEvent::FIX_M, MemoryT::L0C, false>(EVENT_ID0);
    HWaitFlag<HardEvent::FIX_M, MemoryT::L0C, false>(EVENT_ID0);
}

TEST_F(TEST_HSYNC, HSYNC)
{
    HsyncWrapper();
    EXPECT_NO_THROW(GlobalMockObject::verify());
}

TEST_F(TEST_HSYNC, DSB1)
{
    DataSyncBarrier<MemDsbT::ALL>();
}

TEST_F(TEST_HSYNC, DSB)
{
    MOCKER(DataSyncBarrier<MemDsbT::ALL>).expects(once());
    DataSyncBarrier<MemDsbT::ALL>();
    EXPECT_NO_THROW(GlobalMockObject::verify());
}