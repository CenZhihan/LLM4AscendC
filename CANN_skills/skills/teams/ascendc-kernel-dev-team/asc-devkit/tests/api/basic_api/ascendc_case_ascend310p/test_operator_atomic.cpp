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
#include "common.h"
#include "kernel_utils.h"

using namespace std;
using namespace AscendC;

class TEST_ATOMIC_TYPE : public testing::Test {
protected:
    void SetUp()
    {}
    void TearDown()
    {}
};

TEST_F(TEST_ATOMIC_TYPE, ATOMIC_TYPE)
{
    MOCKER(SetAtomicType<float>).expects(once());
    MOCKER(SetAtomicType<half>).expects(once());
    MOCKER(SetAtomicType<int16_t>).expects(once());
    SetAtomicType<float>();
    SetAtomicType<half>();
    SetAtomicType<int16_t>();
    EXPECT_NO_THROW(GlobalMockObject::verify());
}