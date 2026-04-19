/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "mockcpp/mockcpp.hpp"
#include <gtest/gtest.h>
#define private public
#define protected public
#include "kernel_operator.h"
#include "kernel_tensor_impl.h"

using namespace AscendC;

class TEST_POP_STACK_BUFFER : public testing::Test {
    protected:
    // 每一个case之前
    void SetUp() {}
    // 每一个case之后
    void TearDown() {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_POP_STACK_BUFFER, TensorPopStackBuffer)
{
    AscendC::TPipe pipe;
    TQue<TPosition::VECIN, 1> vecInQue;
    GetTPipePtr()->InitBuffer(vecInQue, 1, 128 * sizeof(int32_t));

    LocalTensor<int32_t> tensor;
    int32_t offset = 0;
    uint32_t addr = GetTPipePtr()->GetQueueEndAddress<TPosition::LCM>();
    uint32_t totalAddr = GetEndAddress<TPosition::LCM>();
    LocalTensor<int32_t> stackBuffer;
    bool popRet = PopStackBuffer<int32_t, TPosition::LCM>(stackBuffer);
    EXPECT_EQ(popRet, true);
    LocalTensor<int32_t> temBuffer = stackBuffer[offset];
    EXPECT_EQ(stackBuffer.GetSize(), (totalAddr - addr) / sizeof(int32_t));
}