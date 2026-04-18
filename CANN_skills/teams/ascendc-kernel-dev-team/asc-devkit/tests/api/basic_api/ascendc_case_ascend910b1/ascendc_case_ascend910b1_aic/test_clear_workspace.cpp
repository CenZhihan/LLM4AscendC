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
#include "test_utils.h"
#undef ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/kfc/kernel_kfc.h"
#define ASCENDC_CPU_DEBUG

using namespace std;
using namespace matmul;
using namespace AscendC;

TEST(ClearWorkspace, Case1)
{
    MOCKER(AscendC::SetAtomicNone).expects(once());
    int32_t tmp = g_coreType;
    g_coreType = AscendC::AIC_TYPE;
    constexpr size_t workspaceSize = AscendC::RESERVED_WORKSPACE;
    uint8_t* sysWorkSpacePtr = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    memset(sysWorkSpacePtr, 0, workspaceSize);
    if (sysWorkSpacePtr == nullptr) {
        printf("[error]sysWorkSpacePtr is null, check sysWorkSpacePtr has been set or not\n");
    }
    set_ffts_base_addr(reinterpret_cast<uint64_t>(sysWorkSpacePtr));
    matmul::clearWorkspace(sysWorkSpacePtr);
    AscendC::GmFree((void*)sysWorkSpacePtr);
    g_coreType = tmp;
    EXPECT_NO_THROW(GlobalMockObject::verify());
}