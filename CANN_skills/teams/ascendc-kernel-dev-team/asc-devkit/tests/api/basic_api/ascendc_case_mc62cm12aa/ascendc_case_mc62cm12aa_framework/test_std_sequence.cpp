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
#define private public
#define protect public
#include "kernel_operator.h"
#include <iostream>
 
 
class SequenceTestSuite : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SequenceTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "SequenceTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp()
    {}
    virtual void TearDown()
    {}
};
TEST_F(SequenceTestSuite, SequenceTestCase)
{
    auto seq10 = AscendC::Std::make_index_sequence<10>();
    EXPECT_TRUE((std::is_same_v<decltype(seq10), AscendC::Std::index_sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>>));
}